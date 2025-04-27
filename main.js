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

  if (isDev) {
    // In development, load the Next.js development server
    console.log('Loading development server at http://localhost:3000');
    mainWindow.loadURL('http://localhost:3000');

    // Open DevTools for debugging
    mainWindow.webContents.openDevTools();
  } else {
    // In production, load the built Next.js files
    console.log('Loading production build');
    mainWindow.loadFile(path.join(__dirname, 'ai-assistance-frontend/out/index.html'));
  }
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});