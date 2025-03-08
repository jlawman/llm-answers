'use client';

import { trackEvent } from 'fathom-client';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';

export default function FathomEventsDemo() {
  const handleSimpleEvent = () => {
    trackEvent('simple_event');
  };

  const handleEventWithValue = () => {
    trackEvent('purchase', { _value: 9999 }); // Value in cents
  };

  const handleMultipleProperties = () => {
    trackEvent('event_with_site_id', {
      _value: 15000,
      _site_id: process.env.NEXT_PUBLIC_FATHOM_SITE_ID,
    });
  };

  return (
    <div className="max-w-3xl mx-auto p-8 bg-gray-100 rounded-lg shadow-md">
      <h1 className="text-3xl font-bold mb-6 text-center text-gray-800">Fathom Analytics Event Tracking Demo</h1>
      <p className="mb-6 text-gray-600 text-center">
        Click the buttons below to trigger various Fathom Analytics events. 
        Visit <a href="https://app.usefathom.com" className="text-blue-600 hover:underline" target="_blank" rel="noopener noreferrer">https://app.usefathom.com</a> to view the logged events.
      </p>
      <div className="space-y-8">
        <EventButton
          onClick={handleSimpleEvent}
          color="blue"
          label="Track Simple Event"
          description="Logs a basic event without any additional data."
          code={`import { trackEvent } from 'fathom-client';
trackEvent('simple_event');`}
        />
        <EventButton
          onClick={handleEventWithValue}
          color="green"
          label="Track Purchase Event with Value"
          description="Logs a purchase event with a value of $99.99."
          code={`import { trackEvent } from 'fathom-client'; 
trackEvent('purchase', { _value: 9999 }); // Value in cents`}
        />
        <EventButton
          onClick={handleMultipleProperties}
          color="red"
          label="Track Event with Site ID"
          description="Logs an event value and a site ID."
          code={`import { trackEvent } from 'fathom-client'; trackEvent('event_with_site_id', {
  _value: 15000,
  _site_id: process.env.NEXT_PUBLIC_FATHOM_SITE_ID,
});`}
        />
      </div>
    </div>
  );
}

interface EventButtonProps {
  onClick: () => void;
  color: string;
  label: string;
  description: string;
  code: string;
}

function EventButton({ onClick, color, label, description, code }: EventButtonProps) {
  return (
    <div className="bg-white p-4 rounded-md shadow">
      <button 
        onClick={onClick} 
        className={`w-full px-4 py-2 bg-${color}-500 text-white rounded hover:bg-${color}-600 transition duration-300`}
      >
        {label}
      </button>
      <p className="mt-2 text-sm text-gray-600">{description}</p>
      <div className="mt-4">
        <SyntaxHighlighter language="javascript" style={tomorrow}>
          {code}
        </SyntaxHighlighter>
      </div>
    </div>
  );
}
