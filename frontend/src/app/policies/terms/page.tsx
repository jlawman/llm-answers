import React from 'react';
import { marked } from 'marked';

const termsMarkdown = `
# Terms of Service

*Last updated: November 20, 2023*

From everyone at Adder Analytics, thank you for using Red Panda!

When we say “Company”, “we”, “our”, or “us” in this document, we are referring to Adder Analytics Ltd. 

When we say “Services”, we mean our websitesand any product created and maintained by Adder Analytics Ltd. 

When we say “You” or “your”, we are referring to the people or organizations that own an account or use our Services. 

We may update these Terms of Service ("Terms") in the future. Whenever we make a significant change to our policies, we will refresh the date at the top of this page and take any other appropriate steps to notify account holders.

When you use our Services, now or in the future, you are agreeing to the latest Terms. There may be times where we do not exercise or enforce a right or provision of the Terms; however, that does not mean we are waiving that right or provision. **These Terms do contain a limitation of our liability.**

If you violate any of the Terms, we may terminate your account. 

## Account Terms

1. You are responsible for maintaining the security of your account and password. The Company cannot and will not be liable for any loss or damage from your failure to comply with this security obligation. 
2. You may not use the Services for any purpose outlined in our [Use Restrictions policy](/policies/abuse), and you may not permit any of your users to do so, either.
3. You are responsible for all videos translated and activity that occurs under your account.
4. You must be a human. Accounts registered by “bots” or other automated methods are not permitted.

## Payment, Refunds, and Plan Changes

1. If you are using a trial version of one of our Services, it is really free and — just like for customers who pay for our Services — we do not sell your data.
2. If you are upgrading from a free plan to a paid plan, we will charge your card immediately and your billing cycle starts on the day of upgrade. For other upgrades or downgrades in plan level, the new rate starts from the next billing cycle. You are responsible for payment of all taxes, levies, or duties.
3. We process refunds according to our [Fair Refund policy](/policies/refunds).

## Cancellation and Termination

1. You are solely responsible for properly canceling your account. 
2. All of your content will be inaccessible from the Services immediately upon account cancellation and all related data is subject to deletion.
3. If you cancel the Service before the end of your current paid up month, your cancellation will take effect immediately, and you will not be charged again. We do not automatically prorate unused time in the last billing cycle. See our [Fair Refund policy](policies/refunds) for more details.
4. We have the right to suspend or terminate your account and refuse any and all current or future use of our Services for any reason at any time. Suspension means you and any other users on your account will not be able to access the account or any content in the account. Termination will furthermore result in the deletion of your account or your access to your account, and the forfeiture and relinquishment of all content in your account. We also reserve the right to refuse the use of the Services to anyone for any reason at any time. 

## Modifications to the Service and Prices

1. Sometimes we change the pricing structure for our products. When we do that, we tend to exempt existing customers from those changes. However, we may choose to change the prices for existing customers. If we do so, we will give at least 30 days notice and will notify you via the email address on record. We may also post a notice about changes on our websites or the affected Services themselves.

## Uptime, Security, and Privacy

1. Your use of the Services is at your sole risk. We provide these Services on an “as is” and “as available” basis. We do not offer service-level agreements for most of our Services but do take uptime of our applications seriously.
2. We take many measures to protect and secure your data through backups, redundancies, and encryption. 
3. When you use our Services, you entrust us with your data. We take that trust to heart. You agree that Adder Analytics may process your data as described in our [Privacy Policy](/policies/privacy) and for no other purpose. We as humans can access your data for the following reasons:
  - **To help you with support requests you make.**
  - **On the rare occasions when an error occurs that stops an automated process partway through.** When we can fix the issue and restart automated processing without looking at any personal data, we do. In rare cases, we have to look at a minimum amount of personal data to fix the issue. In these rare cases, we aim to fix the root cause to prevent the errors from recurring.
  - **To safeguard Adder Analytics.** We’ll look at logs and metadata as part of our work to ensure the security of your data and the Services as a whole. 
  - **To the extent required by applicable law.** 

4. We use third party vendors and hosting partners to provide the necessary hardware, software, networking, storage, and related technology required to run the Services.



## Copyright and Content Ownership

1. All content posted on the Services must comply with U.S. copyright law. 
2. You give us a limited license to use the content posted by you in order to provide the Services to you, but we claim no ownership rights over those materials. All materials you submit to the Services remain yours.
3. We do not pre-screen content, but we reserve the right (but not the obligation) in our sole discretion to refuse or remove any content that is available via the Service.
4. The Company or its licensors own all right, title, and interest in and to the Services, including all intellectual property rights therein, and you obtain no ownership rights in the Services as a result of your use. You may not duplicate, copy, or reuse any portion of the HTML, CSS, JavaScript, or visual design elements without express written permission from the Company. You must request permission to use the Company’s logos or any Service logos for promotional purposes. Please [email us](mailto:support@translatethisvideo.com) requests to use logos. We reserve the right to rescind any permissions if you violate these Terms.
5. You agree not to reproduce, duplicate, copy, sell, resell or exploit any portion of the Services, use of the Services, or access to the Services without the express written permission of the Company.

## Features and Bugs

We design our Services with care, based on our own experience and the experiences of customers who share their time and feedback. However, there is no such thing as a service that pleases everybody. We make no guarantees that our Services will meet your specific requirements or expectations.

We also test all of our features extensively before shipping them. As with any software, our Services inevitably have some bugs. We track the bugs reported to us and work through priority ones, especially any related to security or privacy. Not all reported bugs will get fixed and we don’t guarantee completely error-free Services.

## Liability

We mention liability throughout these Terms but to put it all in one section:

***You expressly understand and agree that the Company shall not be liable, in law or in equity, to you or to any third party for any direct, indirect, incidental, lost profits, special, consequential, punitive or exemplary damages, including, but not limited to, damages for loss of profits, goodwill, use, data or other intangible losses (even if the Company has been advised of the possibility of such damages), resulting from: (i) the use or the inability to use the Services; (ii) the cost of procurement of substitute goods and services resulting from any goods, data, information or services purchased or obtained or messages received or transactions entered into through or from the Services; (iii) unauthorized access to or alteration of your transmissions or data; (iv) statements or conduct of any third party on the service; (v) or any other matter relating to these Terms or the Services, whether as a breach of contract, tort (including negligence whether active or passive), or any other theory of liability.***

In other words: choosing to use our Services does mean you are making a bet on us. If the bet does not work out, that’s on you, not us. We do our darnedest to be as safe a bet as possible through careful management of the business; investments in security, infrastructure, and talent. If you choose to use our Services, thank you for betting on us.

> Adapted from the [Basecamp open-source policies](https://github.com/basecamp/policies) / [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
`;


//Within our Services, we provide a simple method of cancellation of any subscriptions found on the [account page](/account). An email request to cancel your account is not automatically considered cancellation. If you need help canceling your subscription, you can always [contact our Support team](mailto:support@translatethisvideo.com).

//If you have a question about any of these Terms, please [contact our Support team](mailto:support@promptpromptprompt.com).

export default function TermsPage() {
  const termsContent = marked(termsMarkdown);

  return (
    <div className="max-w-2xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
      <div className="prose prose-md" dangerouslySetInnerHTML={{ __html: termsContent }} />
    </div>
  );
}
