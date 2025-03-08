'use client'

import { motion } from 'framer-motion'
import { Heading, Lead } from '@/components/ui/text'
import { Divider } from '@/components/ui/divider'
import Link from 'next/link'
import { useState, useEffect } from 'react'

// Interface for route information
interface RouteInfo {
  path: string
  name: string
  type: 'page' | 'api'
  description?: string
}

// Dynamic SiteMap component that builds routes from app directory structure
const SiteMap = () => {
  const [activeTab, setActiveTab] = useState<'pages' | 'api'>('pages')
  const [routes, setRoutes] = useState<{
    pages: RouteInfo[]
    apiRoutes: RouteInfo[]
  }>({ pages: [], apiRoutes: [] })
  const [isLoading, setIsLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [isExpanded, setIsExpanded] = useState(true)
  
  useEffect(() => {
    // Fetch routes from an API endpoint we'll create
    const fetchRoutes = async () => {
      try {
        const response = await fetch('/api/sitemap')
        if (!response.ok) throw new Error('Failed to fetch routes')
        const data = await response.json()
        setRoutes(data)
      } catch (error) {
        console.error('Error fetching routes:', error)
      } finally {
        setIsLoading(false)
      }
    }
    
    fetchRoutes()
  }, [])
  
  // Filter routes based on search term
  const filteredPages = routes.pages.filter(route => 
    route.name.toLowerCase().includes(searchTerm.toLowerCase()) || 
    route.path.toLowerCase().includes(searchTerm.toLowerCase())
  )
  
  const filteredApiRoutes = routes.apiRoutes.filter(route => 
    route.name.toLowerCase().includes(searchTerm.toLowerCase()) || 
    route.path.toLowerCase().includes(searchTerm.toLowerCase()) ||
    (route.description && route.description.toLowerCase().includes(searchTerm.toLowerCase()))
  )
  
  // Group API routes by category (first segment after /api/)
  const groupedApiRoutes = filteredApiRoutes.reduce((acc, route) => {
    // Extract category from path
    const pathParts = route.path.split('/').filter(Boolean);
    const category = pathParts.length > 1 ? pathParts[1] : 'root';
    
    if (!acc[category]) {
      acc[category] = [];
    }
    acc[category].push(route);
    return acc;
  }, {} as Record<string, RouteInfo[]>);
  
  return (
    <div className="bg-white/90 backdrop-blur-sm rounded-lg p-6 shadow-sm">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 gap-4">
        <div className="flex items-center">
          <h3 className="font-semibold text-gray-800 text-xl">Developer Reference</h3>
          <button 
            onClick={() => setIsExpanded(!isExpanded)}
            className="ml-2 p-1 text-gray-500 hover:text-gray-700 focus:outline-none"
            aria-label={isExpanded ? "Collapse content" : "Expand content"}
          >
            <svg 
              xmlns="http://www.w3.org/2000/svg" 
              className={`h-5 w-5 transition-transform duration-200 ${isExpanded ? 'rotate-180' : ''}`} 
              fill="none" 
              viewBox="0 0 24 24" 
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
        </div>
        <div className="flex flex-col sm:flex-row gap-4 w-full sm:w-auto">
          <div className="relative">
            <input
              type="text"
              placeholder="Search routes..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="px-3 py-1 text-sm border border-gray-300 rounded-md w-full sm:w-64 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            {searchTerm && (
              <button
                onClick={() => setSearchTerm('')}
                className="absolute right-2 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
              >
                ×
              </button>
            )}
          </div>
          <div className="flex space-x-2">
            <button 
              onClick={() => setActiveTab('pages')}
              className={`px-3 py-1 text-sm rounded-md transition-colors ${
                activeTab === 'pages' 
                  ? 'bg-blue-100 text-blue-700 font-medium' 
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              Pages ({filteredPages.length})
            </button>
            <button 
              onClick={() => setActiveTab('api')}
              className={`px-3 py-1 text-sm rounded-md transition-colors ${
                activeTab === 'api' 
                  ? 'bg-green-100 text-green-700 font-medium' 
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              API Routes ({filteredApiRoutes.length})
            </button>
          </div>
        </div>
      </div>
      
      {isExpanded && (
        <>
          {isLoading ? (
            <div className="flex justify-center py-8">
              <div className="animate-pulse flex space-x-2">
                <div className="h-2 w-2 bg-gray-400 rounded-full"></div>
                <div className="h-2 w-2 bg-gray-400 rounded-full"></div>
                <div className="h-2 w-2 bg-gray-400 rounded-full"></div>
              </div>
            </div>
          ) : (
            <div className="overflow-x-auto">
              {activeTab === 'pages' ? (
                <>
                  {filteredPages.length > 0 ? (
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Page Name
                          </th>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Path
                          </th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {filteredPages.map((route) => (
                          <tr key={route.path} className="hover:bg-gray-50">
                            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                              {route.name}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-blue-600 font-mono">
                              <Link href={route.path} className="hover:underline">
                                {route.path}
                              </Link>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : (
                    <div className="text-center py-8 text-gray-500">
                      No pages found matching &ldquo;{searchTerm}&rdquo;
                    </div>
                  )}
                </>
              ) : (
                <>
                  {filteredApiRoutes.length > 0 ? (
                    <div className="space-y-6">
                      {Object.entries(groupedApiRoutes).map(([category, routes]) => (
                        <div key={category} className="overflow-hidden">
                          <h4 className="text-sm font-medium text-gray-700 mb-2 px-6 capitalize">
                            {category === 'root' ? 'Root API Routes' : category}
                          </h4>
                          <table className="min-w-full divide-y divide-gray-200">
                            <thead className="bg-gray-50">
                              <tr>
                                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                  API Endpoint
                                </th>
                                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                  Path
                                </th>
                                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                  Description
                                </th>
                              </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-gray-200">
                              {routes.map((route) => (
                                <tr key={route.path} className="hover:bg-gray-50">
                                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                                    {route.name}
                                  </td>
                                  <td className="px-6 py-4 whitespace-nowrap text-sm text-green-600 font-mono">
                                    {route.path}
                                  </td>
                                  <td className="px-6 py-4 text-sm text-gray-500">
                                    {route.description || 'API endpoint'}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-gray-500">
                      No API routes found matching &ldquo;{searchTerm}&rdquo;
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </>
      )}
      
      <div className="mt-4 text-xs text-gray-500 text-right">
        <span>Auto-generated from codebase structure</span>
      </div>
    </div>
  )
}

export default function Home() {
  return (
    <div className="min-h-screen relative overflow-hidden bg-gradient-to-br from-rose-50 via-white to-sky-50">
      <div className="absolute inset-0 bg-grid-slate-100 [mask-image:linear-gradient(0deg,transparent,black)] pointer-events-none" />
      
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="relative max-w-5xl mx-auto px-4 pt-32 sm:pt-40 text-center"
      >
        <Heading 
          className="text-transparent bg-clip-text bg-gradient-to-r from-gray-900 via-gray-800 to-gray-900"
        >
          TIME TO BUILD
        </Heading>

        <Lead className="mt-6 text-gray-600">
          All ideas are ideas.
        </Lead>

        <Divider className="my-12" soft />

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.6 }}
          className="w-full"
        >
          <SiteMap />
        </motion.div>
      </motion.div>

      <motion.div
        className="absolute inset-0 -z-10"
        animate={{
          background: [
            'radial-gradient(circle at 50% 50%, rgba(255,220,220,0.1) 0%, transparent 50%)',
            'radial-gradient(circle at 60% 40%, rgba(220,220,255,0.1) 0%, transparent 50%)',
            'radial-gradient(circle at 40% 60%, rgba(255,220,220,0.1) 0%, transparent 50%)',
          ]
        }}
        transition={{ duration: 8, repeat: Infinity }}
      />
    </div>
  )
}
