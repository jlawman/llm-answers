'use client'

import React from 'react'
import { toast, Toaster } from 'sonner'
import { Button } from '@/components/ui/button'
import { Container } from '@/components/ui/container'
import { Loader2, CheckCircle, XCircle, Bell, ShieldAlert } from 'lucide-react'

export default function ToastDemo() {
  const showDefaultToast = () => toast('This is a default toast')
  const showSuccessToast = () => toast.success('This is a success toast')
  const showErrorToast = () => toast.error('This is an error toast')
  const showInfoToast = () => toast.info('This is an info toast')
  const showWarningToast = () => toast.warning('This is a warning toast')
  const showPromiseToast = () => {
    const promise = new Promise((resolve) => setTimeout(resolve, 2000))
    toast.promise(promise, {
      loading: 'Loading...',
      success: 'Promise resolved!',
      error: 'Promise rejected!',
    })
  }
  const showActionToast = () => toast('Event has been created', {
    action: {
      label: 'Undo',
      onClick: () => console.log('Undo')
    },
  })

  // Custom toasts section
  const showCustomToasts = {
    basic: () => toast.custom(() => (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4 flex items-center justify-between">
        <span className="text-gray-800 dark:text-gray-200">This is a custom toast</span>
        <button 
          onClick={() => toast.dismiss()}
          className="ml-4 px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
        >
          Dismiss
        </button>
      </div>
    )),

    withIcon: () => toast.custom(() => (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4 flex items-center">
        <Bell className="text-blue-500 mr-3" size={24} />
        <span className="text-gray-800 dark:text-gray-200">Custom toast with icon</span>
      </div>
    )),

    withProgress: () => {
      let progress = 0
      const toastId = toast.custom(() => (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4">
          <div className="flex items-center mb-2">
            <ShieldAlert className="text-yellow-500 mr-3" size={24} />
            <span className="text-gray-800 dark:text-gray-200">Updating security...</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
            <div className="bg-yellow-500 h-2.5 rounded-full transition-all duration-500 ease-in-out" style={{ width: `${progress}%` }}></div>
          </div>
        </div>
      ), { duration: Infinity })

      const interval = setInterval(() => {
        progress += 20
        if (progress > 100) {
          clearInterval(interval)
          toast.dismiss(toastId)
          toast.custom(() => (
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4">
              <div className="flex items-center">
                <CheckCircle className="text-green-500 mr-3" size={24} />
                <span className="text-gray-800 dark:text-gray-200">Security update complete!</span>
              </div>
            </div>
          ))
        } else {
          toast.custom(() => (
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4">
              <div className="flex items-center mb-2">
                <ShieldAlert className="text-yellow-500 mr-3" size={24} />
                <span className="text-gray-800 dark:text-gray-200">Updating security...</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
                <div className="bg-yellow-500 h-2.5 rounded-full transition-all duration-500 ease-in-out" style={{ width: `${progress}%` }}></div>
              </div>
            </div>
          ), { id: toastId })
        }
      }, 1000)
    },

    customPromise: () => {
      const promise = new Promise((resolve, reject) => {
        setTimeout(Math.random() > 0.5 ? resolve : reject, 2000)
      })

      toast.promise(promise, {
        loading: (
          <div className="flex items-center">
            <Loader2 className="animate-spin mr-2" size={18} />
            <span>Processing payment...</span>
          </div>
        ),
        success: (
          <div className="flex items-center">
            <CheckCircle className="text-green-500 mr-2" size={18} />
            <span>Payment successful!</span>
          </div>
        ),
        error: (
          <div className="flex items-center">
            <XCircle className="text-red-500 mr-2" size={18} />
            <span>Payment failed. Please try again.</span>
          </div>
        ),
      })
    }
  }

  return (
    <Container className="py-16">
      <h1 className="text-4xl font-bold mb-8">Sonner Toast Demo</h1>
      
      {/* Reference link */}
      <p className="mb-8">
        For more information, check out the <a href="https://sonner.emilkowal.ski/" target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">Sonner documentation</a>.
      </p>
      
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-4">Basic Toasts</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Button onClick={showDefaultToast} className="bg-gray-500 hover:bg-gray-600">Default Toast</Button>
          <Button onClick={showSuccessToast} className="bg-green-500 hover:bg-green-600">Success Toast</Button>
          <Button onClick={showErrorToast} className="bg-red-500 hover:bg-red-600">Error Toast</Button>
          <Button onClick={showInfoToast} className="bg-blue-500 hover:bg-blue-600">Info Toast</Button>
          <Button onClick={showWarningToast} className="bg-yellow-500 hover:bg-yellow-600 text-black">Warning Toast</Button>
          <Button onClick={showPromiseToast} className="bg-purple-500 hover:bg-purple-600">Promise Toast</Button>
          <Button onClick={showActionToast} className="bg-indigo-500 hover:bg-indigo-600">Action Toast</Button>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-4">Custom Toasts</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Button onClick={showCustomToasts.basic} className="bg-pink-400 hover:bg-pink-500">Basic Custom Toast</Button>
          <Button onClick={showCustomToasts.withIcon} className="bg-cyan-400 hover:bg-cyan-500">Custom Toast with Icon</Button>
          <Button onClick={showCustomToasts.withProgress} className="bg-yellow-400 hover:bg-yellow-500 text-black">Progress Toast</Button>
          <Button onClick={showCustomToasts.customPromise} className="bg-violet-400 hover:bg-violet-500">Custom Promise Toast</Button>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-4">Toast Positions</h2>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          {(['top-left', 'top-center', 'top-right', 'bottom-left', 'bottom-center', 'bottom-right'] as const).map((position) => (
            <Button 
              key={position} 
              onClick={() => toast(`Toast position: ${position}`, { position })}
              className="bg-teal-400 hover:bg-teal-500"
            >
              {position}
            </Button>
          ))}
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">Toast Appearance</h2>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <Button 
            onClick={() => toast('Rich colors toast', { style: { background: 'linear-gradient(to right, #00b09b, #96c93d)' } })}
            className="bg-gradient-to-r from-green-300 to-blue-400 hover:from-green-400 hover:to-blue-500"
          >
            Rich Colors Toast
          </Button>
          <Button 
            onClick={() => toast('Headless toast', { unstyled: true })}
            className="bg-gray-200 text-white-700 hover:bg-gray-300"
          >
            Headless Toast
          </Button>
        </div>
      </section>

      <Toaster expand={false} richColors />
    </Container>
  )
}
