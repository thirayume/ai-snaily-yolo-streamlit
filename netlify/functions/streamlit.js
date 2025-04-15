const { spawn } = require('child_process');
const path = require('path');

exports.handler = async function(event, context) {
  // Path to your Streamlit app
  const appPath = path.join(__dirname, '../../app.py');
  
  // Run Streamlit
  const streamlit = spawn('streamlit', ['run', appPath]);
  
  // Log output
  streamlit.stdout.on('data', (data) => {
    console.log(`stdout: ${data}`);
  });
  
  streamlit.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
  });
  
  // Return response
  return {
    statusCode: 200,
    body: JSON.stringify({ message: "Streamlit app is running" }),
  };
};