// import { StrictMode } from 'react'
// import { createRoot } from 'react-dom/client'
// import './index.css'
// import App from './App.jsx'

// createRoot(document.getElementById('root')).render(
//   <StrictMode>
//     <App />
//   </StrictMode>,
// )

import React from "react";
import { createRoot } from "react-dom/client";
import ChatWidget from "./ChatWidget";
import "./index.css";

// Dynamically create a container for the widget
const widgetContainerId = "chat-widget-root";
let widgetContainer = document.getElementById(widgetContainerId);

if (!widgetContainer) {
  widgetContainer = document.createElement("div");
  widgetContainer.id = widgetContainerId;
  document.body.appendChild(widgetContainer);
}

// Render the ChatWidget directly into the container
createRoot(widgetContainer).render(<ChatWidget />);
