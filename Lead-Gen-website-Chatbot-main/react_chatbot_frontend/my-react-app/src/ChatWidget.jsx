import React, { useState, useEffect } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faPaperPlane } from "@fortawesome/free-solid-svg-icons";
import ReactMarkdown from "react-markdown";
import "./ChatWidget.css";

const ChatWidget = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [userMessage, setUserMessage] = useState("");
  const [isTyping, setIsTyping] = useState(false);

  const toggleChat = () => {
    const chatBubble = document.querySelector(".chat-bubble");

    if (isOpen) {
      chatBubble.classList.add("closing");
      setTimeout(() => {
        setIsOpen(false);
        chatBubble.classList.remove("closing");
      }, 300);
    } else {
      chatBubble.classList.add("opening");
      setTimeout(() => {
        setIsOpen(true);
        chatBubble.classList.remove("opening");
      }, 300);
    }
  };

  useEffect(() => {
    if (isOpen && messages.length === 0) {
      setMessages([{ sender: "bot", text: "Hey, I'm Nebula9.ai's chatbot. How can I help you?" }]);
    }
  }, [isOpen, messages]);

  const typeWriterEffect = (text, delay = 100) => {
    return new Promise((resolve) => {
      const words = text.split(" ");
      let currentText = "";
      const interval = setInterval(() => {
        if (words.length > 0) {
          currentText += (currentText ? " " : "") + words.shift();
          setMessages((prev) => [
            ...prev.slice(0, -1),
            { sender: "bot", text: currentText },
          ]);
        } else {
          clearInterval(interval);
          resolve();
        }
      }, delay);
    });
  };

  const sendMessage = async () => {
    if (!userMessage.trim()) return;

    const newMessages = [...messages, { sender: "user", text: userMessage }];
    setMessages(newMessages);
    setUserMessage("");

    setIsTyping(true);

    try {
      const response = await fetch("http://localhost:8000/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userMessage }),
      });

      const data = await response.json();
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "" }, // Placeholder for typewriter effect
      ]);

      setIsTyping(false);

      await typeWriterEffect(data.response, 100);
      setIsTyping(false);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "Error fetching response!" },
      ]);
      setIsTyping(false);
    }
  };

  return (
    <div className="chat-widget">
      <div className="chat-bubble" onClick={toggleChat}>
        {isOpen ? "✖" : "🗪"}
      </div>

      {isOpen && (
        <div className="chat-box">
          <div className="chat-header">Nebula9.ai Chatbot</div>
          <div className="chat-messages">
            {messages.map((msg, index) => (
              <div
                key={index}
                className={`message ${msg.sender === "user" ? "user" : "bot"}`}
              >
                {msg.sender === "bot" ? (
                  <ReactMarkdown
                    children={msg.text}
                    components={{
                      a: ({ href, children }) => (
                        <a
                          href={href}
                          target="_blank"
                          rel="noopener noreferrer"
                          style={{
                            color: "#1e90ff",
                            textDecoration: "underline",
                          }}
                        >
                          {children}
                        </a>
                      ),
                    }}
                  />
                ) : (
                  <span>{msg.text}</span>
                )}
              </div>
            ))}
            {isTyping && (
              <div className="message bot typing">
                <span>.</span>
                <span>.</span>
                <span>.</span>
              </div>
            )}
          </div>
          <div className="chat-input">
            <input
              type="text"
              placeholder="Type your message..."
              value={userMessage}
              onChange={(e) => setUserMessage(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && sendMessage()}
            />
            <button className="send-button" onClick={sendMessage}>
              <FontAwesomeIcon icon={faPaperPlane} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatWidget;
