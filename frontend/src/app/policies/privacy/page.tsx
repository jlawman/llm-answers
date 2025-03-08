import React from 'react';
import { marked } from 'marked';

const termsMarkdown = `
# Privacy policy

*Last updated: October 18, 2024*

The privacy of your data—and it is your data, not ours!—is a big deal to us. In this policy, we lay out: what data we collect and why; how your data is handled; and your rights with respect to your data. We promise we never sell your data: never have, never will.

This policy applies to all products built and maintained by Adder Analytics Ltd including all versions of Red Panda. 

This policy applies to our handling of information about site visitors, prospective customers, and customers and authorized users (in relation to their procurement of the services and management of their relationship with Adder Analytics). We refer collectively to these categories of individuals as "you" throughout this policy.

## What we collect and why

Our guiding principle is to collect only what we need. Here&apos;s what that means in practice:

### Identity and access

When you sign up for an Adder Analytics product, we ask for your email address. That&apos;s so you can identify your new account, and we can send you product updates and other essential information. We may also send you optional surveys from time to time to help us understand how you use our products and to make improvements. With your consent, we will send you our newsletter and other updates. 

We&apos;ll never sell your personal information to third parties, and we won&apos;t use your e-mail address, YouTube Channel, or other personal information in marketing statements without your permission either.

### Billing information

If you sign up for a paid Adder Analytics product, you will be asked to provide your payment information. Credit card information is submitted directly to our payment processor and doesn’t hit Adder Analytics servers. 

### Product interactions

We store on our servers the content that you upload or receive or maintain in your Red Panda account. This is so that we can offer our services and so that you can download the content related to your translations. This content is usually stored for 7 days. 


### Voluntary correspondence

When you email our support team with a question or to ask for help, we keep that correspondence, including your email address, so that we have a history of past correspondence to reference if you reach out in the future.

We also store information you may volunteer, for example, written responses to surveys. If you agree to a customer interview, we may ask for your permission to record the conversation for future reference or use. We will only do so with your express consent.

## When we access or disclose your information

**To provide products or services you&apos;ve requested**. We use some third-party subprocessors to help run our applications and provide the Services to you. 

**To help troubleshoot or squash a software bug** 

**To investigate, prevent, or take action regarding [restricted uses](policies/abuse).** Accessing a customer&apos;s account when investigating potential abuse is a measure of last resort. We want to protect the privacy and safety of both our customers and the people reporting issues to us, and we do our best to balance those responsibilities throughout the process. If we discover you are using our products for a restricted purpose, we will take action as necessary, including notifying appropriate authorities where warranted.

**Aggregated and de-identified data.** We may aggregate and/or de-identify information collected through the services. We may use de-identified or aggregated data for any purpose, including marketing or analytics.

**To improve the quality of the services offered** We occasionally re-translate videos with an updated process and analyze the updated version in comparison with the original in order to improve the quality of our product. 

**When required under applicable law.** Adder Analytics is incorporated in the UK.

Finally, if Adder Analytics is acquired by or merges with another company we&apos;ll notify you well before any of your personal information is transferred or becomes subject to a different privacy policy.


## Data retention

We keep your information for the time necessary for the purposes for which it is processed. The length of time for which we retain information depends on the purposes for which we collected and use it and your choices, after which time we may delete and/or aggregate it. We may also retain and use this information as necessary to comply with our legal obligations, resolve disputes, and enforce our agreements. Through this policy, we have provided specific retention periods for certain types of information.

## Location of site and data

Our products and other web properties have operations in the United States. If you are located in the European Union, UK, or elsewhere outside of the United States, **please be aware that any information you provide to us may be transferred to and stored in the United States**. By using our websites or Services and/or providing us with your personal information, you consent to this transfer.

## Changes and questions

We may update this policy as needed to comply with relevant regulations and reflect any new practices. Whenever we make a significant change to our policies, we will refresh the date at the top of this page and take any other appropriate steps to notify users.

> Adapted from the [Basecamp open-source policies](https://github.com/basecamp/policies) / [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
`;

export default function TermsPage() {
  const termsContent = marked(termsMarkdown);

  return (
    <div className="max-w-3xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
      <div className="prose prose-md" dangerouslySetInnerHTML={{ __html: termsContent }} />
    </div>
  );
}
