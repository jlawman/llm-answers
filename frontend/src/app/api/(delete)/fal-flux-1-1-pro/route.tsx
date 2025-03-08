import { NextRequest, NextResponse } from 'next/server';
import { fal } from "@fal-ai/client";

fal.config({
  credentials: process.env.FAL_API_KEY
});

interface FluxInput {
  prompt: string;
  image_size?: string;
  num_inference_steps?: number;
  guidance_scale?: number;
  num_images?: number;
  enable_safety_checker?: boolean;
  strength?: number;
  output_format?: string;
}

export async function POST(request: NextRequest) {
  console.log('POST request received');
  try {
    const { 
      prompt, 
      image_size, 
      num_inference_steps, 
      guidance_scale, 
      num_images, 
      enable_safety_checker, 
      strength, 
      output_format 
    } = await request.json();

    console.log('Request body:', { prompt, image_size, num_inference_steps, guidance_scale, num_images, enable_safety_checker, strength, output_format });

    if (!prompt) {
      console.log('Error: Prompt is missing');
      return NextResponse.json({ error: 'Prompt is required' }, { status: 400 });
    }

    // Add validation for num_images
    if (num_images !== undefined && (typeof num_images !== 'number' || num_images > 1 || num_images < 1)) {
      console.log('Error: Invalid num_images value');
      return NextResponse.json({ error: 'num_images must be exactly 1' }, { status: 400 });
    }

    const input: FluxInput = { prompt };

    // Add optional parameters if they are provided
    if (image_size) input.image_size = image_size;
    if (num_inference_steps) input.num_inference_steps = num_inference_steps;
    if (guidance_scale) input.guidance_scale = guidance_scale;
    if (num_images) input.num_images = num_images;
    if (enable_safety_checker !== undefined) input.enable_safety_checker = enable_safety_checker;
    if (strength) input.strength = strength;
    if (output_format) input.output_format = output_format;

    console.log('Prepared input for fal.subscribe:', input);

    const result = await fal.subscribe("fal-ai/flux-pro/v1.1", {
      input,
      logs: true,
      onQueueUpdate: (update) => {
        console.log('Queue update:', update);
        if (update.status === "IN_PROGRESS" && 'logs' in update) {
          update.logs.map((log) => log.message).forEach(console.log);
        }
      },
    });

    console.log('fal.subscribe result:', result);

    return NextResponse.json({ 
      data: result.data,
      requestId: result.requestId 
    });

  } catch (error: unknown) {
    console.error('Error:', error);
    if (error && typeof error === 'object' && 'status' in error && 'body' in error) {
      const typedError = error as { status: number; body: { detail?: string } };
      if (typedError.status === 422 && typedError.body && typedError.body.detail) {
        // Handle validation error
        console.log('Validation Error:', typedError.body.detail);
        return NextResponse.json({ 
          error: 'Validation Error', 
          details: typedError.body.detail 
        }, { status: 422 });
      }
    }
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}
