// app/providers.js
'use client'
import posthog from 'posthog-js'
import { PostHogProvider } from 'posthog-js/react'
import mixpanel from 'mixpanel-browser'
import { createContext, useContext, useEffect } from 'react'

if (typeof window !== 'undefined') {
  posthog.init(process.env.NEXT_PUBLIC_POSTHOG_KEY, {
    api_host: process.env.NEXT_PUBLIC_POSTHOG_HOST,
    person_profiles: 'identified_only', // or 'always' to create profiles for anonymous users as well
  })

  // Initialize Mixpanel
  mixpanel.init(process.env.NEXT_PUBLIC_MIXPANEL_PROJECT_TOKEN, {
    debug: true,
    track_pageview: true,
    persistence: 'localStorage',
  })
}

// Create Mixpanel context
const MixpanelContext = createContext(null)

export function CSPostHogProvider({ children }) {
    return <PostHogProvider client={posthog}>{children}</PostHogProvider>
}

export function MixpanelProvider({ children }) {
  useEffect(() => {
    if (typeof window !== 'undefined') {
      // Initialize Mixpanel here
      mixpanel.init(process.env.NEXT_PUBLIC_MIXPANEL_TOKEN, {
        debug: true,
        track_pageview: true,
        persistence: 'localStorage',
      });
    }
  }, []);

  return (
    <MixpanelContext.Provider value={mixpanel}>
      {children}
    </MixpanelContext.Provider>
  )
}

export const useMixpanel = () => useContext(MixpanelContext)
