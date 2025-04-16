exports.handler = async function(event, context) {
  return {
    statusCode: 200,
    headers: {
      "Content-Type": "text/html",
    },
    body: `
    <!DOCTYPE html>
    <html>
      <head>
        <title>YOLO Object Detection</title>
        <style>
          body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            text-align: center;
          }
          .container {
            max-width: 800px;
            margin: 0 auto;
          }
          h1 {
            color: #333;
          }
          .message {
            background-color: #f8f9fa;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 20px;
            margin-top: 20px;
          }
          .links {
            margin-top: 30px;
          }
          .links a {
            display: inline-block;
            margin: 10px;
            padding: 12px 20px;
            background-color: #ff4b4b;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
          }
          .links a:hover {
            background-color: #e04040;
          }
        </style>
      </head>
      <body>
        <div class="container">
          <h1>YOLO Object Detection App</h1>
          <div class="message">
            <p>This application requires a persistent server to run Streamlit.</p>
            <p>Netlify Functions are not suitable for hosting Streamlit applications due to their serverless nature and timeout limitations.</p>
          </div>
          <div class="links">
            <a href="https://share.streamlit.io/" target="_blank">Try Streamlit Cloud</a>
            <a href="https://github.com/thirayume/ai-snaily-yolo-streamlit" target="_blank">View Source on GitHub</a>
          </div>
        </div>
      </body>
    </html>
    `
  };
};