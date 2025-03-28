import { useState } from "react";
import { ChatBotWidget } from "chatbot-widget-ui";

const App = () => {
  // Save all messages in conversation
  const [messages, setMessages] = useState<
    { type: string; text: string }[]
  >([]);

  // Function to call your backend /chat endpoint
  const customApiCall = async (message: string): Promise<string> => {
    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: message }), // Match your API's input structure
      });

      if (!response.ok) {
        throw new Error("Failed to fetch");
      }

      const data = await response.json();

      // Extract the 'response' field from the API response
      return data.response; // Ensure this matches your API structure
    } catch (error) {
      console.error("Error:", error);
      return "Sorry, something went wrong. Please try again.";
    }
  };

  const handleNewMessage = async (userMessage: string) => {
    // Add the user's message to the chat
    setMessages((prevMessages) => [
      ...prevMessages,
      { type: "user", text: userMessage },
    ]);

    // Get the bot's response
    const botResponse = await customApiCall(userMessage);

    // Add the bot's response to the chat
    setMessages((prevMessages) => [
      ...prevMessages,
      { type: "bot", text: botResponse },
    ]);
  };

  return (
    <div>
      <ChatBotWidget
        callApi={customApiCall}
        primaryColor="#7161EF"
        inputMsgPlaceholder="Type your message..."
        chatbotName="Nebula9.ai Chatbot"
        isTypingMessage="Typing..."
        IncommingErrMsg="Oops! Something went wrong. Try again."
        handleNewMessage={handleNewMessage} // Correctly handle new messages
        chatIcon={<div>O</div>}
      />
    </div>
  );
};

export default App;