from typing import Dict, Any, List, Tuple, Optional

from llama_index.llama_pack.base import BaseLlamaPack
from llama_index.agent.types import BaseAgent
from .gradio_theme import llama

from ansi2html import Ansi2HTMLConverter

from io import StringIO
import sys


class Capturing(list):
    """To capture the stdout from ReActAgent.chat with verbose=True. Taken from
    https://stackoverflow.com/questions/16571150/\
        how-to-capture-stdout-output-from-a-python-function-call
    """

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout


class GradioAgentChatPack(BaseLlamaPack):
    """Gradio chatbot to chat with your own Agent."""

    def __init__(
        self,
        agent: BaseAgent,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        self.agent = agent
        self.thoughts = ""
        self.conv = Ansi2HTMLConverter()

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {"agent": self.agent}

    def _handle_user_message(self, user_message, history):
        """Handle the user submitted message. Clear message box, and append
        to the history."""
        return "", history + [(user_message, "")]

    def _generate_response(self, chat_history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
        """Generate the response from agent, and capture the stdout of the
        ReActAgent's thoughts.
        """
        with Capturing() as output:
            response = self.agent.stream_chat(chat_history[-1][0])
        ansi = "\n========\n".join(output)
        html_output = self.conv.convert(ansi)
        for token in response.response_gen:
            chat_history[-1][1] += token
            yield chat_history, str(html_output)

    def _reset_chat(self) -> Tuple[str, str]:
        """Reset the agent's chat history. And clear all dialogue boxes."""
        # clear agent history
        self.agent.reset()
        return "", "", ""  # clear textboxes

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        import gradio as gr

        demo = gr.Blocks(
            theme=llama,
            css="#box { height: 420px; overflow-y: scroll !important}"
        )
        with demo:
            gr.Markdown(
                "# Gradio ReActAgent Powered by LlamaIndex and LlamaHub 🦙\n"
                "This Gradio app is powered by LlamaIndex's `ReActAgent` with\n"
                "OpenAI's GPT-4-Turbo as the LLM. The tools are listed below.\n"
            )
            with gr.Row():
                chat_window = gr.Chatbot(
                    label="Message History",
                    scale=3,
                )
                console = gr.HTML(elem_id="box")
            with gr.Row():
                message = gr.Textbox(label="Write A Message", scale=4)
                clear = gr.ClearButton()

            message.submit(
                self._handle_user_message,
                [message, chat_window],
                [message, chat_window],
                queue=False
            ).then(
                self._generate_response,
                chat_window,
                [chat_window, console],
            )
            clear.click(
                self._reset_chat,
                None,
                [message, chat_window, console]
            )

        demo.launch(server_name="0.0.0.0", server_port=8080)
