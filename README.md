# ONE-TIME SETUP

1) Create required API keys and add to new Doppler project

    - Dopper project name "red-panda-simple"
    - Doppler secrets:
    - BRAINTRUST_API_KEY
    - DATABASE_URL (Neon)
    - FATHOM_API_KEY
    - ANTHROPIC_API_KEY
    - Any other LLM keys

 



# Create a new Red Panda project

```bash
bash -c 'cd ~/Documents/adder/"100 templates"/red-panda-simple && ./setup/red-panda-creator.sh'
```


# Update database

```bash
npx drizzle-kit generate
npx drizzle-kit push
```
