import React from 'react'

interface Tab {
  name: string
  current: boolean
}

interface TabsProps {
  tabs: Tab[]
  onChange: (selectedTab: Tab) => void
}

function classNames(...classes: string[]) {
  return classes.filter(Boolean).join(' ')
}

export function Tabs({ tabs, onChange }: TabsProps) {
  return (
    <div className="border-b border-gray-200 overflow-x-auto scrollbar-hide">
      <nav className="-mb-px flex space-x-8 min-w-full sm:min-w-0" aria-label="Tabs">
        {tabs.map((tab) => (
          <button
            key={tab.name}
            onClick={() => onChange(tab)}
            className={classNames(
              tab.current
                ? 'border-indigo-500 text-indigo-600'
                : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700',
              'whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm flex-shrink-0'
            )}
            aria-current={tab.current ? 'page' : undefined}
          >
            {tab.name}
          </button>
        ))}
      </nav>
    </div>
  )
}
