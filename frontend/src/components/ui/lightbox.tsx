import React, { useState } from 'react';
import Image from 'next/image';
import { X } from 'lucide-react';

const Lightbox = ({ src, alt, width, height }: {
  src: string;
  alt: string;
  width: number;
  height: number;
}) => {
  const [isOpen, setIsOpen] = useState(false);

  const openLightbox = () => setIsOpen(true);
  const closeLightbox = () => setIsOpen(false);

  return (
    <>
      <div className="cursor-pointer transition-transform duration-300 hover:scale-105">
        <Image
          src={src}
          alt={alt}
          width={width}
          height={height}
          onClick={openLightbox}
          layout="responsive"
        />
      </div>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-90 transition-opacity duration-300">
          <button
            onClick={closeLightbox}
            className="absolute top-4 right-4 text-white hover:text-gray-300"
          >
            <X size={24} />
          </button>
          <div className="relative h-[90vh] w-[90vw]">
            <Image
              src={src}
              alt={alt}
              layout="fill"
              objectFit="contain"
              quality={100}
            />
          </div>
        </div>
      )}
    </>
  );
};

export default Lightbox;
