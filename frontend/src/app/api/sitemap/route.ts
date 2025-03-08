import { NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'

// Interface for route information
interface RouteInfo {
  path: string
  name: string
  type: 'page' | 'api'
  description?: string
}

// Function to clean route name by removing text in brackets
function cleanRouteName(name: string): string {
  // Remove text in parentheses
  return name.replace(/\([^)]*\)/g, '').trim()
    .replace(/-/g, ' ')
    .replace(/^\w/, c => c.toUpperCase())
    .replace(/\s+/g, ' ') // Remove extra spaces
}

// Function to clean route path by removing parentheses
function cleanRoutePath(path: string): string {
  // Split the path into segments
  const segments = path.split('/').filter(Boolean);
  
  // Clean each segment by removing parentheses
  const cleanedSegments = segments.map(segment => 
    segment.replace(/[()]/g, '')
  );
  
  // Reconstruct the path
  return cleanedSegments.length > 0 ? '/' + cleanedSegments.join('/') : '/';
}

// Function to recursively scan the app directory and find all page files
function scanAppDirectory(dir: string, basePath = '', type: 'page' | 'api' = 'page'): RouteInfo[] {
  const appDir = path.join(process.cwd(), dir)
  let routes: RouteInfo[] = []
  
  if (!fs.existsSync(appDir)) return routes
  
  const entries = fs.readdirSync(appDir, { withFileTypes: true })
  
  for (const entry of entries) {
    const entryPath = path.join(dir, entry.name)
    
    // Skip layout files and other special files
    if (
      entry.name.startsWith('_') || 
      entry.name.startsWith('.') || 
      entry.name === 'layout.tsx' ||
      entry.name === 'loading.tsx' ||
      entry.name === 'error.tsx' ||
      entry.name === 'not-found.tsx'
    ) {
      continue
    }
    
    if (entry.isDirectory()) {
      // For API routes
      if (type === 'page' && entry.name === 'api') {
        routes = [...routes, ...scanAppDirectory(entryPath, '/api', 'api')]
        continue
      }
      
      // Handle route groups (directories in parentheses)
      let routeSegment = '';
      
      if (entry.name.startsWith('(') && entry.name.endsWith(')')) {
        // For route groups like (demos), don't add to path but scan contents
        routeSegment = '';
      } else if (entry.name.startsWith('(')) {
        // For partial route groups, extract the name without parentheses
        const cleanName = entry.name.replace(/[()]/g, '');
        routeSegment = `/${cleanName}`;
      } else {
        routeSegment = `/${entry.name}`;
      }
      
      const newBasePath = basePath + routeSegment;
      
      // Recursively scan subdirectories
      routes = [...routes, ...scanAppDirectory(entryPath, newBasePath, type)]
    } else if (
      type === 'page' &&
      (entry.name === 'page.tsx' || entry.name === 'page.js')
    ) {
      // Found a page file, create a route
      const routePath = cleanRoutePath(basePath) || '/';
      
      // Extract a more readable name from the path
      let routeName = '';
      
      if (routePath === '/') {
        routeName = 'Home';
      } else {
        // Get the last segment of the path
        const lastSegment = routePath.split('/').pop() || '';
        routeName = cleanRouteName(lastSegment);
      }
      
      routes.push({ 
        path: routePath, 
        name: routeName,
        type: 'page'
      });
    } else if (
      type === 'api' &&
      (entry.name === 'route.ts' || entry.name === 'route.js')
    ) {
      // Found an API route file
      const routePath = cleanRoutePath(basePath);
      
      // Extract a more readable name from the path
      let routeName = '';
      
      if (routePath === '/api') {
        routeName = 'API Root';
      } else {
        // Get the last segment of the path
        const lastSegment = routePath.split('/').pop() || '';
        routeName = cleanRouteName(lastSegment);
      }
      
      routes.push({ 
        path: routePath, 
        name: routeName,
        type: 'api',
        description: getRouteDescription(entryPath)
      });
    }
  }
  
  return routes;
}

// Function to extract a simple description from the route file
function getRouteDescription(filePath: string): string {
  try {
    const content = fs.readFileSync(path.join(process.cwd(), filePath), 'utf8')
    
    // Look for comments that might describe the API
    const commentMatch = content.match(/\/\*\*([\s\S]*?)\*\//) || content.match(/\/\/(.*?)$/m)
    if (commentMatch) {
      return commentMatch[1].trim().replace(/\*/g, '').replace(/\n/g, ' ').trim()
    }
    
    // Look for HTTP methods
    const methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'].filter(method => 
      content.includes(`export async function ${method}`) || 
      content.includes(`export function ${method}`)
    )
    
    if (methods.length) {
      return `Supports ${methods.join(', ')} methods`
    }
    
    return 'API endpoint'
  } catch (error) {
    return 'API endpoint'
  }
}

// Function to scan all directories including those in (delete)
function scanAllDirectories(): RouteInfo[] {
  // Start with regular app directory scan
  let routes = scanAppDirectory('src/app');
  
  // Scan the (delete) directory for pages
  const deleteDir = path.join(process.cwd(), 'src/app/(delete)');
  if (fs.existsSync(deleteDir)) {
    const entries = fs.readdirSync(deleteDir, { withFileTypes: true });
    
    for (const entry of entries) {
      if (entry.isDirectory()) {
        // Scan each subdirectory in (delete) as if it were at the root
        const subDirPath = path.join(deleteDir, entry.name);
        const cleanName = cleanRouteName(entry.name);
        const cleanPath = cleanRoutePath(`/${entry.name}`);
        
        // Check if it has a page.tsx
        if (fs.existsSync(path.join(subDirPath, 'page.tsx')) || 
            fs.existsSync(path.join(subDirPath, 'page.js'))) {
          routes.push({
            path: cleanPath,
            name: cleanName,
            type: 'page'
          });
        }
        
        // Recursively scan this directory
        const subRoutes = scanAppDirectory(
          path.join('src/app/(delete)', entry.name), 
          cleanPath
        );
        routes = [...routes, ...subRoutes];
      }
    }
  }
  
  // Scan the api/(delete) directory for API routes
  const apiDeleteDir = path.join(process.cwd(), 'src/app/api/(delete)');
  if (fs.existsSync(apiDeleteDir)) {
    const entries = fs.readdirSync(apiDeleteDir, { withFileTypes: true });
    
    for (const entry of entries) {
      if (entry.isDirectory()) {
        // Check if it has a route.ts or route.js
        const apiDirPath = path.join(apiDeleteDir, entry.name);
        const cleanName = cleanRouteName(entry.name);
        const cleanPath = cleanRoutePath(`/api/${entry.name}`);
        
        if (fs.existsSync(path.join(apiDirPath, 'route.ts')) || 
            fs.existsSync(path.join(apiDirPath, 'route.js'))) {
          routes.push({
            path: cleanPath,
            name: cleanName,
            type: 'api',
            description: 'API endpoint'
          });
        }
        
        // Recursively scan this directory
        const subRoutes = scanAppDirectory(
          path.join('src/app/api/(delete)', entry.name), 
          cleanPath,
          'api'
        );
        routes = [...routes, ...subRoutes];
      }
    }
  }
  
  return routes;
}

export async function GET() {
  try {
    // Get all routes including those in (delete) directories
    const routes = scanAllDirectories()
    
    // Group routes by type
    const pages = routes.filter(route => route.type === 'page')
    const apiRoutes = routes.filter(route => route.type === 'api')
    
    // Remove duplicates based on path
    const uniquePages = Array.from(
      new Map(pages.map(page => [page.path, page])).values()
    )
    
    const uniqueApiRoutes = Array.from(
      new Map(apiRoutes.map(api => [api.path, api])).values()
    )
    
    // Sort routes alphabetically, with home at the top for pages
    uniquePages.sort((a, b) => {
      if (a.path === '/') return -1
      if (b.path === '/') return 1
      return a.path.localeCompare(b.path)
    })
    
    // Sort API routes alphabetically
    uniqueApiRoutes.sort((a, b) => a.path.localeCompare(b.path))
    
    return NextResponse.json({ 
      pages: uniquePages,
      apiRoutes: uniqueApiRoutes
    })
  } catch (error) {
    console.error('Error generating sitemap:', error)
    return NextResponse.json(
      { error: 'Failed to generate sitemap' },
      { status: 500 }
    )
  }
} 