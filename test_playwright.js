const { chromium } = require('playwright');

(async () => {
  console.log('Starting Playwright test...');
  
  try {
    // Launch browser
    const browser = await chromium.launch({ 
      headless: true  // Run in headless mode for speed
    });
    
    // Create a new page
    const page = await browser.newPage();
    
    // Set page content directly (no network required)
    await page.setContent(`
      <html>
        <head>
          <title>Playwright Test Page</title>
        </head>
        <body>
          <h1>Hello from Playwright!</h1>
          <p>This is a test page created locally.</p>
        </body>
      </html>
    `);
    
    // Get page title
    const title = await page.title();
    console.log('Page title:', title);
    
    // Get heading text
    const heading = await page.textContent('h1');
    console.log('Heading text:', heading);
    
    // Take a screenshot
    await page.screenshot({ path: 'test-screenshot.png' });
    console.log('Screenshot saved as test-screenshot.png');
    
    // Close browser
    await browser.close();
    
    console.log('Test completed successfully!');
  } catch (error) {
    console.error('Test failed:', error);
    process.exit(1);
  }
})();