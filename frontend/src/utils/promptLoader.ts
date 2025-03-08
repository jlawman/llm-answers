import fs from 'fs/promises';
import path from 'path';
import yaml from 'js-yaml';

async function listDirectoryContents(dirPath: string) {
  try {
    const files = await fs.readdir(dirPath);
    console.log(`Contents of ${dirPath}:`);
    files.forEach(file => {
      console.log(file);
    });
  } catch (error) {
    console.error(`Error reading directory ${dirPath}:`, error);
  }
}

export async function loadPrompt(promptName: string): Promise<string> {
  const promptsDir = path.join(process.cwd(), 'src/prompts');
  const promptFilePath = path.join(promptsDir, `${promptName}.yaml`);
  
  console.log(`Current working directory: ${process.cwd()}`);
  await listDirectoryContents(promptsDir);
  
  console.log(`Attempting to load prompt from: ${promptFilePath}`);

  try {
    const promptYaml = await fs.readFile(promptFilePath, 'utf8');
    console.log(`Successfully read YAML file: ${promptName}.yaml`);

    const promptData = yaml.load(promptYaml) as { prompt: string };

    return promptData.prompt;
  } catch (error) {
    console.error(`Error loading prompt ${promptName}:`, error);
    throw error;
  }
}
