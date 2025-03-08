'use client'

import { useState } from 'react'
import { Radio, RadioGroup } from '@headlessui/react'
import { CheckCircleIcon } from '@heroicons/react/20/solid'

interface Option {
  id: number
  title: string
  description: string
  additionalInfo: string
}

interface RadioSelectCardsProps {
  options: Option[]
  legend: string
  additionalInfoLabel: string
  onChange: (option: Option) => void
}

export default function RadioSelectCards({ options, legend, additionalInfoLabel, onChange }: RadioSelectCardsProps) {
  const [selectedOption, setSelectedOption] = useState(options[0])

  const handleChange = (option: Option) => {
    setSelectedOption(option)
    onChange(option)
  }

  return (
    <fieldset>
      <legend className="text-sm font-semibold leading-6 text-gray-900">{legend}</legend>
      <RadioGroup
        value={selectedOption}
        onChange={handleChange}
        className="mt-6 grid grid-cols-1 gap-y-6 sm:grid-cols-3 sm:gap-x-4"
      >
        {options.map((option) => (
          <Radio
            key={option.id}
            value={option}
            aria-label={option.title}
            aria-describedby={`description-${option.id}`}
            className="group relative flex cursor-pointer rounded-lg border border-gray-300 bg-white p-4 shadow-sm focus:outline-none data-[focus]:border-indigo-600 data-[focus]:ring-2 data-[focus]:ring-indigo-600"
          >
            <span className="flex flex-1">
              <span className="flex flex-col">
                <span className="block text-sm font-medium text-gray-900">{option.title}</span>
                <span className="mt-1 flex items-center text-sm text-gray-500">{option.description}</span>
                <span className="mt-6 text-sm font-medium text-gray-900">{option.additionalInfo}</span>
              </span>
            </span>
            <CheckCircleIcon
              aria-hidden="true"
              className="h-5 w-5 text-indigo-600 [.group:not([data-checked])_&]:invisible"
            />
            <span
              aria-hidden="true"
              className="pointer-events-none absolute -inset-px rounded-lg border-2 border-transparent group-data-[focus]:border group-data-[checked]:border-indigo-600"
            />
            <span id={`description-${option.id}`} className="sr-only">
              {option.description} {additionalInfoLabel} {option.additionalInfo}
            </span>
          </Radio>
        ))}
      </RadioGroup>
    </fieldset>
  )
}
