//To update the db run:
//npx drizzle-kit push

//To review tables run:
//npx drizzle-kit studio

import { pgTable, text, timestamp } from 'drizzle-orm/pg-core'

export const UserMessages = pgTable('user_messages', {
  user_id: text('user_id').primaryKey().notNull(),
  createTs: timestamp('create_ts').defaultNow().notNull(),
  message: text('message').notNull(),
})

export const ProductHuntEntries = pgTable('product_hunt_entries', {
  id: text('id').primaryKey().notNull(),
  name: text('name').notNull(),
  description: text('description').notNull(),
  url: text('url').notNull(),
  publishDate: text('publish_date').notNull(),
  author: text('author').notNull(),
  upvotes: text('upvotes').notNull(),
  createTs: timestamp('create_ts').defaultNow().notNull(),
})

export const AINews = pgTable('ai_news', {
  id: text('id').primaryKey().notNull(),
  title: text('title').notNull(),
  published: text('published').notNull(),
  summary: text('summary').notNull(),
  createTs: timestamp('create_ts').defaultNow().notNull(),
})

export const IndustryGroups = pgTable('industry_groups', {
  industryGroup: text('industry_group').primaryKey(),
  definition: text('definition').notNull(),
  industries: text('industries').notNull(),
})

export const BusinessIdeas = pgTable('business_ideas', {
  id: text('id').primaryKey().notNull(),
  industryGroup: text('industry_group').notNull(),
  industry: text('industry').notNull(),
  prompt: text('prompt').notNull(),
  analysis: text('analysis').notNull(),
  idea: text('idea').notNull(),
  name: text('name').notNull(),
  description: text('description').notNull(),
  createTs: timestamp('create_ts').defaultNow().notNull(),
})

export const UserGeneratedBusinessIdeas = pgTable('user_generated_business_ideas', {
  id: text('id').primaryKey().notNull(),
  industryGroup: text('industry_group').notNull(),
  industry: text('industry').notNull(),
  prompt: text('prompt').notNull(),
  analysis: text('analysis').notNull(),
  idea: text('idea').notNull(),
  name: text('name').notNull(),
  description: text('description').notNull(),
  createTs: timestamp('create_ts').defaultNow().notNull(),
})