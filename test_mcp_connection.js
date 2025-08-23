// Test MCP connection and configuration
const { spawn } = require('child_process');

console.log('Testing Playwright MCP connection...');

// Try to spawn playwright-mcp
const mcp = spawn('npx', ['-y', 'playwright-mcp'], {
    shell: true,
    stdio: ['pipe', 'pipe', 'pipe']
});

let output = '';

mcp.stdout.on('data', (data) => {
    output += data.toString();
    console.log('MCP Output:', data.toString());
});

mcp.stderr.on('data', (data) => {
    console.error('MCP Error:', data.toString());
});

mcp.on('error', (error) => {
    console.error('Failed to start MCP:', error);
});

mcp.on('close', (code) => {
    console.log(`MCP process exited with code ${code}`);
    if (output.includes('MCP Server started') || output.includes('server started')) {
        console.log('✅ Playwright MCP appears to be working!');
    } else {
        console.log('❌ Playwright MCP may not be working correctly');
    }
});

// Send test initialization after 2 seconds
setTimeout(() => {
    console.log('Sending initialization request...');
    const initRequest = {
        jsonrpc: '2.0',
        method: 'initialize',
        params: {
            protocolVersion: '1.0.0',
            clientInfo: {
                name: 'test-client',
                version: '1.0.0'
            }
        },
        id: 1
    };
    
    mcp.stdin.write(JSON.stringify(initRequest) + '\n');
}, 2000);

// Terminate after 5 seconds
setTimeout(() => {
    console.log('Terminating test...');
    mcp.kill();
    process.exit(0);
}, 5000);