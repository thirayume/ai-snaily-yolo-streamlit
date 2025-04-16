const { spawn } = require('child_process');
const path = require('path');

exports.handler = async function(event, context) {
  // Start Streamlit process
  const streamlitProcess = spawn('streamlit', ['run', 'app.py']);
  
  // Collect any output from Streamlit
  let output = '';
  streamlitProcess.stdout.on('data', (data) => {
    output += data.toString();
  });
  
  // Handle Streamlit process completion
  await new Promise((resolve) => {
    streamlitProcess.on('close', (code) => {
      resolve();
    });
  });
  
  return {
    statusCode: 200,
    body: output || "Streamlit app started"
  };
};