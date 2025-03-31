const { app, BrowserWindow } = require('electron');
const path = require('path');
const isDev = process.env.NODE_ENV === 'development';

function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true, // Needed for preload.js (if you use it)
      contextIsolation: false, // Consider security implications
      // More web preferences can be added here
    },
  });

  if (isDev) {
    // In development, load the Next.js development server
    mainWindow.loadURL('http://localhost:3000'); // Default Next.js port
    mainWindow.webContents.openDevTools(); // Open DevTools for debugging
  } else {
    // In production, load the built Next.js files
    // You'll need to run `npm run build` in your ai-assistant-frontend directory first
    mainWindow.loadFile(path.join(__dirname, 'ai-assistant-frontend/out/index.html'));
  }
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});