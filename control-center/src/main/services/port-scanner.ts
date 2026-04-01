import * as net from 'net';

/**
 * Check if a TCP port is currently in use on 127.0.0.1.
 * Returns true if the port is occupied, false if free.
 */
export function isPortInUse(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const server = net.createServer();

    server.once('error', (err: NodeJS.ErrnoException) => {
      if (err.code === 'EADDRINUSE') {
        resolve(true);
      } else {
        // Other error (permission denied, etc.) — treat as in-use to be safe
        resolve(true);
      }
    });

    server.once('listening', () => {
      server.close(() => {
        resolve(false);
      });
    });

    server.listen(port, '127.0.0.1');
  });
}
