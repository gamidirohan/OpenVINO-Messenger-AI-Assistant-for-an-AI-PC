const { app, BrowserWindow, shell } = require('electron');
const path = require('path');
const isDev = process.env.NODE_ENV !== 'production';

function createWindow() {
  // Create the browser window
  const mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    icon: path.join(__dirname, 'ai-assistance-frontend/public/favicon.ico'),
    title: 'OpenVINO Messenger AI Assistant'
  });

  // Set window title
  mainWindow.setTitle('OpenVINO Messenger AI Assistant');

  // Open external links in the default browser instead of Electron
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });

  // Always load from the Next.js development server in this version
  console.log('Loading development server at http://localhost:3000');

  // Try to load the Next.js development server
  mainWindow.loadURL('http://localhost:3000').catch(err => {
    console.error('Failed to load Next.js development server:', err);

    // Show an error message in the Electron window
    mainWindow.loadURL(`data:text/html,
      <html>
        <head>
          <title>Error</title>
          <style>
            body { font-family: Arial, sans-serif; padding: 20px; color: #333; }
            h1 { color: #e53e3e; }
            pre { background: #f8f8f8; padding: 10px; border-radius: 5px; }
          </style>
        </head>
        <body>
          <h1>Failed to connect to frontend server</h1>
          <p>Make sure the Next.js development server is running at <strong>http://localhost:3000</strong></p>
          <p>Run the following command in the ai-assistance-frontend directory:</p>
          <pre>npm run dev</pre>
          <p>Error details:</p>
          <pre>${err.toString()}</pre>
        </body>
      </html>
    `);
  });

  // Open DevTools for debugging
  mainWindow.webContents.openDevTools();
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});