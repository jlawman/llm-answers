'use client'

import { motion } from 'framer-motion'
import { Heading, Lead } from '@/components/ui/text'
import { Divider } from '@/components/ui/divider'

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
        >
     
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
