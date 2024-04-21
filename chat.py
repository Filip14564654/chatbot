import tkinter as tk
from ollama_test import ChatBot  # Assuming your ChatBot class is in a file named 'chatbot.py'

class ChatWindow:
    def __init__(self, master):
        self.master = master
        master.title("Chat Window")

        # Create an instance of the ChatBot class
        self.chat_bot = ChatBot(model_name="llama3")

        self.chat_log = tk.Text(master, state='disabled')
        self.chat_log.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.entry_field = tk.Entry(master)
        self.entry_field.bind("<Return>", self.send_message)
        self.entry_field.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        self.send_button = tk.Button(master, text="Send", command=self.send_message)
        self.send_button.grid(row=1, column=1, padx=10, pady=10, sticky="e")

    def send_message(self, event=None):
        message = self.entry_field.get()
        response = self.chat_bot.ask_question(message)  # Call ask_question on the chat_bot instance
        self.entry_field.delete(0, tk.END)
        self.display_message("User: " + message)
        self.display_message("Bot: " + response)

    def display_message(self, message):
        self.chat_log.configure(state='normal')
        self.chat_log.insert(tk.END, message + '\n')
        self.chat_log.configure(state='disabled')
        self.chat_log.see(tk.END)

def main():
    root = tk.Tk()
    chat_window = ChatWindow(root)
    root.mainloop()

if __name__ == "__main__":
    main()
