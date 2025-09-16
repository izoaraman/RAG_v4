import gradio as gr


class UISettings:
    """
    Utility class for managing UI settings.

    This class provides static methods for toggling UI components, such as a sidebar.
    """
    @staticmethod
    def toggle_sidebar(state):
        """
        Toggle the visibility state of a UI component.

        Parameters:
            state: The current state of the UI component.

        Returns:
            Tuple: A tuple containing the updated UI component state and the new state.
        """
        state = not state
        return gr.update(visible=state), state

    @staticmethod
    def feedback(data: gr.LikeData):
        if data.liked:
            print("You upvoted this response: " + data.value)
        else:
            print("You downvoted this response: " + data.value)

    @staticmethod
    def chatbot():
        return gr.Chatbot(type="messages")

    @staticmethod
    def title():
        title = """
        <div style="display: flex; align-items: center; justify-content: center; color: #984EA3;">
            <img src="images/ai.jpg" alt="Chatbot Icon" style="height: 60px; margin-right: 10px;">
            <h1 style="font-size: 3rem; margin: 0;">RAG-Chat</h1>
        </div>
        """
        return title
