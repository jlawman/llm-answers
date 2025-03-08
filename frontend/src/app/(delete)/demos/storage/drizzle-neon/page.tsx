import { createUserMessage, deleteUserMessage } from '@/app/actions'
import { db } from '@/app/db'
import { auth } from '@clerk/nextjs/server'
import { Input } from '@/components/ui/input'
import clsx from 'clsx'

export default async function Home() {
  const { userId } = auth()
  if (!userId) throw new Error('User not found')
  const existingMessage = await db.query.UserMessages.findFirst({
    where: (messages, { eq }) => eq(messages.user_id, userId),
  })

  return (
    <main className="max-w-2xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6">Neon + Clerk Example</h1>
      {existingMessage ? (
        <div className="bg-white dark:bg-zinc-800 rounded-lg p-4 shadow-md">
          <p className="mb-4">{existingMessage.message}</p>
          <form action={deleteUserMessage}>
            <Button variant="danger">Delete Message</Button>
          </form>
        </div>
      ) : (
        <form action={createUserMessage} className="space-y-4">
          <Input type="text" name="message" placeholder="Enter a message" />
          <Button type="submit">Save Message</Button>
        </form>
      )}
    </main>
  )
}

function Button({
  className,
  variant = 'primary',
  ...props
}: React.ComponentPropsWithoutRef<'button'> & {
  variant?: 'primary' | 'danger'
}) {
  return (
    <button
      {...props}
      className={clsx(
        className,
        'inline-flex items-center justify-center rounded-lg px-4 py-2 text-sm font-semibold',
        'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2',
        variant === 'primary' &&
          'bg-blue-600 text-white hover:bg-blue-500 focus-visible:outline-blue-600',
        variant === 'danger' &&
          'bg-red-600 text-white hover:bg-red-500 focus-visible:outline-red-600'
      )}
    />
  )
}
