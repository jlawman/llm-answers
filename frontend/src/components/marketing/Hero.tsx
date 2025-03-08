import { Button } from '@/components/ui/button'
import { Container } from '@/components/ui/container'
import { Gradient } from '@/components/ui/gradient'
import { Link } from '@/components/ui/link'
import { Navbar } from '@/components/ui/navbar'

interface HeroProps {
  show: boolean;
  bannerText: string;
  bannerLink: string;
  heading: string;
  subheading: string;
  theme: 'A' | 'B';
  variant: 'withLink' | 'withoutLink';
}

export function Hero({
  show,
  bannerText,
  bannerLink,
  heading,
  subheading,
  theme = 'A',
  variant = 'withLink',
}: HeroProps) {
  if (!show) return null;

  const renderBanner = () => {
    if (variant === 'withoutLink') return null;
    return (
      <Link
        href={bannerLink}
        className="flex items-center gap-1 rounded-full bg-fuchsia-950/35 px-3 py-0.5 text-sm/6 font-medium text-white data-[hover]:bg-fuchsia-950/30"
      >
        {bannerText}
      </Link>
    );
  };

  const renderThemeContent = () => {
    switch (theme) {
      case 'A':
        return (
          <>
            <div className="pb-24 pt-16 sm:pb-32 sm:pt-24 md:pb-48 md:pt-32">
              <h1 className="font-display text-balance text-6xl/[0.9] font-medium tracking-tight text-gray-950 sm:text-8xl/[0.8] md:text-9xl/[0.8]">
                {heading}
              </h1>
              <p className="mt-8 max-w-lg text-xl/7 font-medium text-gray-950/75 sm:text-2xl/8">
                {subheading}
              </p>
              <div className="mt-12 flex flex-col gap-x-6 gap-y-4 sm:flex-row">
                <Button href="/business-ideas">Find a New SaaS Idea</Button>
                <Button href="/brand" variant="secondary">Define a Brand</Button>
              </div>
            </div>
          </>
        );
      default:
        return null;
    }
  };

  return (
    <div className="relative">
      <Gradient className="absolute inset-2 bottom-0 rounded-4xl ring-1 ring-inset ring-black/5" />
      <Container className="relative">
        <Navbar banner={renderBanner()} />
        {renderThemeContent()}
      </Container>
    </div>
  );
}
