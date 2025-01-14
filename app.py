from pathlib import Path
from typing import List, Tuple, Optional, Set
import json
import dataclasses
import gradio as gr
import asyncio
from openai import AsyncOpenAI
import tempfile
import os
import argparse

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("API_KEY environment variable is not set")

if not BASE_URL:
    print("Using the default base URL (OpenAI)")

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)


@dataclasses.dataclass
class Example:
    input: str
    output: str


@dataclasses.dataclass
class Task:
    entrypoint: str
    examples: List[Example]
    signature: str
    name: str


@dataclasses.dataclass
class TaskResponse:
    completion: str
    # List of booleans that indicate whether or not the test succeeded
    successes: List[bool]
    # List of outputs on each test
    outputs: List[str]


async def run_command(cmd, timeout=5):
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        return (
            stdout.decode("utf-8", errors="ignore"),
            stderr.decode("utf-8", errors="ignore"),
            process.returncode,
        )
    except asyncio.TimeoutError:
        process.kill()
        return None, None, None


def extract_code(response: str) -> str:
    # Find the first code block in the response
    code_block_start = response.find("```")
    if code_block_start == -1:
        return response  # Return the original response if no code block is found
    code_block_end = response.find("```", code_block_start + 3)
    if code_block_end == -1:
        return (
            response  # Return the original response if no closing code block is found
        )
    code_block = response[code_block_start + 3 : code_block_end]
    # Remove the name of the language from the start of the code block if it exists.
    language_start = code_block.find("\n")
    if language_start != -1:
        code_block = code_block[language_start + 1 :]
    return code_block


async def completion(prompt: str, model: str, is_chat: bool) -> str:
    if is_chat:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Complete each given function and give the response in a code block. No need for explanations or tests.",
                },
                {"role": "user", "content": f"Complete this:\n\n```\n{prompt}\n```"},
            ],
            temperature=0.2,
        )
        return extract_code(response.choices[0].message.content)
    else:
        response = await client.completions.create(
            model=model,
            prompt=prompt,
            temperature=0.2,
            max_tokens=512,
            stop=["\ndef", "\nclass", "\n#", "\n@"],
        )
        return response.choices[0].text


def load_tasks(file_path: Path) -> List[Task]:
    tasks = []
    with open(file_path, "r") as f:
        for line in f:
            task_data = json.loads(line)
            task = Task(**task_data)
            task.examples = [Example(**ex) for ex in task.examples]
            tasks.append(task)
    return tasks


async def submit_task(
    task: Task, description: str, username: str, model: str, is_chat: bool
) -> TaskResponse:
    prompt = f'{task.signature}\n    """\n    {description.strip()}\n    """'
    response = await completion(prompt, model, is_chat)
    # Chat models will return a complete function wit the prompt.
    test_program = prompt + response if not is_chat else response
    results = []
    outputs = []
    for example in task.examples:
        prog = (
            test_program
            + f"\n\nout = {task.entrypoint}({example.input})\nprint(repr(out))\nassert out == {example.output}\n"
        )
        with tempfile.NamedTemporaryFile(delete=True, suffix=".py") as f:
            f.write(prog.encode("utf-8"))
            f.flush()
            stdout, stderr, exit_code = await run_command(["python", f.name], timeout=5)
            # Append to the file results_username.jsonl and store
            #  username, task.name, description, response, stdout, stderr, and exit_code
            with open(f"results_{username}.jsonl", "a") as f:
                f.write(
                    json.dumps(
                        {
                            "username": username,
                            "task": task.name,
                            "description": description,
                            "response": response,
                            "stdout": stdout,
                            "stderr": stderr,
                            "exit_code": exit_code,
                        }
                    )
                    + "\n"
                )
            results.append(exit_code == 0)
            # Confirmed that "".split produces a non-empty list.
            outputs.append(stdout.split("\n", maxsplit=1)[0])

    return TaskResponse(response, results, outputs)


def format_task_details(
    task: Task,
    interface: "TaskInterface",
    test_results: Optional[List[bool]] = None,
    test_outputs: Optional[List[str]] = None,
) -> str:
    md = f"### Task: {task.name}\n\n"
    if not interface.description_editable:
        if not interface.is_chat:
            md += f'```python\n{task.signature}\n    """\n    {interface.current_description}\n    """{interface.completion}\n```\n\n'
        else:
            md += f"```python\n{interface.completion}\n```\n\n"
    else:
        md += f"```python\n{task.signature}\n```\n\n"
    md += "### Examples:\n\n"
    md += "| Input | Output |"
    if test_results is not None:
        md += " Test Results |"
    if test_outputs is not None:
        md += " Actual Outputs |"
    md += "\n"
    md += "|-------|--------|"
    if test_results is not None:
        md += "-------------|"
    if test_outputs is not None:
        md += "-------------|"
    md += "\n"
    for i, example in enumerate(task.examples):
        md += f"| {example.input} | {example.output} |"
        if test_results is not None:
            result = "✅" if test_results[i] else "❌"
            md += f" {result} |"
        if test_outputs is not None:
            md += f" {test_outputs[i]} |"
        md += "\n"
    return md


@dataclasses.dataclass
class TaskInterface:
    tasks: List[Task]
    model: str
    is_chat: bool
    current_index: int = 0
    current_description: str = ""
    description_editable: bool = True
    completion: str = ""
    username: str = ""

    def get_current_task(self) -> Task:
        return self.tasks[self.current_index]

    def next_task(self) -> str:
        self.current_index += 1
        self.current_description = ""  # Reset description for new task
        self.completion = ""
        if self.current_index >= len(self.tasks):
            return "Completed"
        return f"Task: {self.get_current_task().name}"

    async def process_submission(self, description: str) -> TaskResponse:
        self.current_description = description
        task = self.get_current_task()
        return await submit_task(
            task, description, self.username, self.model, self.is_chat
        )

    def update_description(self, description: str) -> None:
        self.current_description = description

    def set_description_editable(self, editable: bool) -> None:
        self.description_editable = editable


def on_login(users: Set[str]):
    def callback(username, interface: TaskInterface):
        if username not in users:
            return (
                gr.update(visible=True),  # login still visible
                gr.update(visible=False),  # main interface still not visible
                gr.update(visible=True, value="Username not found"),
                gr.update(visible=False),  # keep download button hidden
            )

        interface.username = username
        return (
            gr.update(visible=False),  # Hide login row
            gr.update(visible=True),  # Show main interface
            "",
            gr.update(visible=True),  # Show download button
        )

    return callback


def create_interface(users: Set[str], tasks: List[Task], model: str, is_chat: bool):
    with gr.Blocks() as demo:
        interface_state = gr.State(TaskInterface(tasks, model=model, is_chat=is_chat))

        # Add login components
        with gr.Column(visible=True) as login_row:

            username_input = gr.Textbox(label="Username")
            login_button = gr.Button("Login")
            login_error_message = gr.Markdown(visible=False)

        # Wrap existing components in a group to control visibility
        with gr.Column(visible=False) as main_interface:
            task_details = gr.Markdown()
            description_input = gr.Textbox(label="Enter task description")
            submit_button = gr.Button("Submit")
            processing_message = gr.Textbox(value="Processing...", visible=False)
            with gr.Row():
                retry_button = gr.Button("Retry", visible=False)
                next_button = gr.Button("Next Task", visible=False)
                download_button = gr.Button("Download Results", visible=False)
            file_to_download = gr.File(visible=False, label="Download Results")

        login_button.click(
            on_login(users),
            inputs=[username_input, interface_state],
            outputs=[login_row, main_interface, login_error_message, download_button],
        )

        # Add download functionality
        def get_download_link(state: TaskInterface):
            file_path = Path(f"results_{state.username}.jsonl")
            if not file_path.exists():
                return None
            return gr.File(value=str(file_path), visible=True)

        download_button.click(
            get_download_link, inputs=[interface_state], outputs=[file_to_download]
        )

        async def on_submit(description, interface: TaskInterface):
            interface.update_description(description)
            interface.set_description_editable(False)
            current_task = interface.get_current_task()

            # Show processing message and hide other elements
            yield (
                gr.update(visible=True),  # Show processing message
                gr.update(visible=False),  # Hide submit button
                gr.update(visible=False),  # Hide retry button
                gr.update(visible=False),  # Hide next button
                gr.update(visible=False),  # Hide description input
                gr.update(
                    visible=True, value=format_task_details(current_task, interface)
                ),  # Update and show task details
            )

            # Process the submission
            task_response = await interface.process_submission(description)
            interface.completion = task_response.completion

            # Hide processing message and show result
            yield (
                gr.update(visible=False),  # Hide processing message
                gr.update(visible=False),  # Hide submit button
                gr.update(visible=True),  # Show retry button
                gr.update(visible=True),  # Show next button
                gr.update(visible=False),  # Hide description input
                gr.update(
                    visible=True,
                    value=format_task_details(
                        current_task,
                        interface,
                        task_response.successes,
                        task_response.outputs,
                    ),
                ),  # Update and show task details
            )

        def on_next(interface: TaskInterface):
            interface.set_description_editable(True)
            next_task = interface.next_task()
            if next_task == "Completed":
                return (
                    gr.update(value="All tasks completed!"),
                    gr.update(visible=False),  # Hide description input
                    gr.update(visible=False),  # Hide submit button
                    gr.update(visible=False),  # Hide processing message
                    gr.update(visible=False),  # Hide retry button
                    gr.update(visible=False),  # Hide next button
                )
            current_task = interface.get_current_task()
            return (
                gr.update(value=format_task_details(current_task, interface)),
                gr.update(value="", visible=True),  # Clear and show description input
                gr.update(visible=True),  # Show submit button
                gr.update(visible=False),  # Hide processing message
                gr.update(visible=False),  # Hide retry button
                gr.update(visible=False),  # Hide next button
            )

        def on_retry(interface: TaskInterface):
            interface.set_description_editable(True)
            return (
                gr.update(visible=True),  # Show description input
                gr.update(visible=True),  # Show submit button
                gr.update(visible=False),  # Hide processing message
                gr.update(visible=False),  # Hide retry button
                gr.update(visible=False),  # Hide next button
            )

        # Update function calls to include interface_state
        submit_button.click(
            on_submit,
            inputs=[description_input, interface_state],
            outputs=[
                processing_message,
                submit_button,
                retry_button,
                next_button,
                description_input,
                task_details,
            ],
        )
        next_button.click(
            on_next,
            inputs=interface_state,
            outputs=[
                task_details,
                description_input,
                submit_button,
                processing_message,
                retry_button,
                next_button,
            ],
        )
        retry_button.click(
            on_retry,
            inputs=interface_state,
            outputs=[
                description_input,
                submit_button,
                processing_message,
                retry_button,
                next_button,
            ],
        )

        # Add this to initialize the task details
        demo.load(
            lambda interface: format_task_details(
                interface.get_current_task(), interface
            ),
            inputs=interface_state,
            outputs=task_details,
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--users", type=str, default="users.txt", help="Path to the users file"
    )
    parser.add_argument(
        "--tasks", type=str, default="tasks.jsonl", help="Path to the tasks file"
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="localhost",
        help="Server name for Gradio interface",
    )
    parser.add_argument(
        "--server-port", type=int, default=7860, help="Port for Gradio interface"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Enable sharing of Gradio interface",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B",
        help="Model name for completions",
    )
    parser.add_argument(
        "--is-chat",
        action="store_true",
        default=False,
        help="Use chat completion instead of regular completion",
    )
    args = parser.parse_args()

    tasks = load_tasks(Path(args.tasks))
    users = Path(args.users).read_text().splitlines()
    users = set(user.strip() for user in users if user.strip())
    demo = create_interface(users, tasks, args.model, args.is_chat)

    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
