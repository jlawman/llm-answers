import { Navbar } from '@/components/ui/navbar'
import { Container } from '@/components/ui/container'

export default function BusinessIdeasLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <>
      <Container>
        <Navbar banner={null} />
      </Container>
      {children}
    </>
  )
}
