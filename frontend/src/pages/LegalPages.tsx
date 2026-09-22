import { Link } from 'react-router-dom'
import type { ReactNode } from 'react'

const UPDATED = '22 September 2026'
const CONTACT = 'info@starlightlinearled.com'
const COMPANY = 'Starlight Linear LED'
const PRODUCT = 'Starlight AI Mailer'

function LegalShell({
  title,
  subtitle,
  children,
  active,
}: {
  title: string
  subtitle: string
  children: ReactNode
  active: 'privacy' | 'terms'
}) {
  return (
    <div className="legal-screen">
      <header className="legal-top">
        <Link to="/login" className="legal-brand">
          <div className="brand-mark">S</div>
          <div>
            <div className="brand"><span>Starlight AI Mailer</span></div>
            <div className="muted" style={{ fontSize: 12, fontWeight: 600 }}>Linear LED</div>
          </div>
        </Link>
        <nav className="legal-tabs">
          <Link to="/privacy" className={active === 'privacy' ? 'active' : ''}>Privacy Policy</Link>
          <Link to="/terms" className={active === 'terms' ? 'active' : ''}>Terms of Service</Link>
          <Link to="/login" className="legal-signin">Sign in</Link>
        </nav>
      </header>

      <article className="legal-doc panel">
        <p className="legal-updated muted">Last updated {UPDATED}</p>
        <h1>{title}</h1>
        <p className="legal-lead">{subtitle}</p>
        <div className="legal-body">{children}</div>
        <footer className="legal-foot">
          <p className="muted">
            Questions? Contact {COMPANY} at{' '}
            <a href={`mailto:${CONTACT}`}>{CONTACT}</a>
          </p>
          <div className="row" style={{ gap: 12 }}>
            <Link to="/privacy">Privacy Policy</Link>
            <Link to="/terms">Terms of Service</Link>
            <Link to="/login">Back to sign in</Link>
          </div>
        </footer>
      </article>
    </div>
  )
}

export function PrivacyPolicyPage() {
  return (
    <LegalShell
      active="privacy"
      title="Privacy Policy"
      subtitle={`How ${COMPANY} collects, uses, and protects information when you use ${PRODUCT}.`}
    >
      <section>
        <h2>1. Who we are</h2>
        <p>
          {COMPANY} (“Starlight”, “we”, “us”) operates {PRODUCT}, an AI-assisted sales and
          outreach platform for LED lighting professionals. This policy explains what data we
          process when you create an account, connect Google / Gmail, upload catalogues or leads,
          and send or receive email through the service.
        </p>
      </section>

      <section>
        <h2>2. Information we collect</h2>
        <ul>
          <li>
            <strong>Account data</strong> — name, email address, password hash (if you use email
            sign-in), organisation name, and role.
          </li>
          <li>
            <strong>Google / Gmail data</strong> — when you sign in with Google or connect Gmail,
            we receive your Google account email and OAuth tokens needed to send mail and sync
            replies on your behalf. We do not store your Google password.
          </li>
          <li>
            <strong>Workspace content</strong> — lead lists, scraped website summaries, email
            drafts, sent mail metadata, conversation threads, catalogues, product images, and
            prompts or templates you configure.
          </li>
          <li>
            <strong>Technical data</strong> — approximate usage logs, IP address, browser type,
            and diagnostics needed to operate and secure the service.
          </li>
        </ul>
      </section>

      <section>
        <h2>3. How we use information</h2>
        <ul>
          <li>Authenticate you and manage your organisation workspace.</li>
          <li>Generate personalised outreach emails and product recommendations.</li>
          <li>Send email and sync inbox replies using the Gmail account you connect.</li>
          <li>Host digital catalogues and attach branded product sheets when you enable that option.</li>
          <li>Improve reliability, prevent abuse, and provide support.</li>
        </ul>
        <p>
          We use Azure OpenAI (or similar AI providers you configure) to generate email content
          and analyse catalogues. Content you submit for generation may be processed by those
          providers under their terms and our configuration.
        </p>
      </section>

      <section>
        <h2>4. Legal bases (where applicable)</h2>
        <p>
          Where data-protection law requires a legal basis, we rely on: performance of a contract
          (providing the service you signed up for), legitimate interests (securing and improving
          the product), and consent (for example Google OAuth scopes you approve).
        </p>
      </section>

      <section>
        <h2>5. Sharing</h2>
        <p>We do not sell your personal data. We share data only with:</p>
        <ul>
          <li>Infrastructure and AI providers that host or process the service for us.</li>
          <li>Google, when you authorise Gmail / Sign in with Google.</li>
          <li>Professionals who help us operate the business (under confidentiality obligations).</li>
          <li>Authorities when required by law.</li>
        </ul>
      </section>

      <section>
        <h2>6. Retention</h2>
        <p>
          We keep account and workspace data while your organisation uses {PRODUCT}. You may
          request deletion of your account or specific content by contacting us. OAuth tokens are
          removed when you disconnect Gmail or delete your integration. Backups may persist for a
          limited period before purge.
        </p>
      </section>

      <section>
        <h2>7. Security</h2>
        <p>
          We use encryption in transit, access controls, and (where configured) encrypted storage
          for integration credentials. No method of transmission or storage is 100% secure; please
          use strong passwords and protect your Google account.
        </p>
      </section>

      <section>
        <h2>8. Your choices</h2>
        <ul>
          <li>Disconnect Gmail from Settings at any time.</li>
          <li>Update sender profile and organisation settings in the app.</li>
          <li>Request access, correction, or deletion of personal data by emailing {CONTACT}.</li>
        </ul>
      </section>

      <section>
        <h2>9. International transfers</h2>
        <p>
          Servers or subprocessors may be located outside your country. Where required, we use
          appropriate safeguards for cross-border transfers.
        </p>
      </section>

      <section>
        <h2>10. Children</h2>
        <p>{PRODUCT} is intended for business users and is not directed to children under 16.</p>
      </section>

      <section>
        <h2>11. Changes</h2>
        <p>
          We may update this policy from time to time. The “Last updated” date at the top will
          change when we do. Continued use of the service after an update means you accept the
          revised policy.
        </p>
      </section>

      <section>
        <h2>12. Contact</h2>
        <p>
          {COMPANY}<br />
          Email: <a href={`mailto:${CONTACT}`}>{CONTACT}</a><br />
          Website: <a href="https://www.starlightlinearled.com" target="_blank" rel="noreferrer">starlightlinearled.com</a>
        </p>
      </section>
    </LegalShell>
  )
}

export function TermsOfServicePage() {
  return (
    <LegalShell
      active="terms"
      title="Terms of Service"
      subtitle={`The rules for using ${PRODUCT}, operated by ${COMPANY}.`}
    >
      <section>
        <h2>1. Agreement</h2>
        <p>
          By accessing or using {PRODUCT} (“Service”), you agree to these Terms of Service
          (“Terms”). If you use the Service on behalf of an organisation, you represent that you
          have authority to bind that organisation.
        </p>
      </section>

      <section>
        <h2>2. The Service</h2>
        <p>
          {PRODUCT} helps sales teams scrape prospect websites, ground outreach in product
          catalogues, draft and send email (including via Google / Gmail), sync replies, and manage
          related workspace content. Features may change as we improve the product.
        </p>
      </section>

      <section>
        <h2>3. Accounts</h2>
        <ul>
          <li>You must provide accurate account information and keep credentials secure.</li>
          <li>You are responsible for activity under your account and organisation workspace.</li>
          <li>
            Google sign-in and Gmail connection are optional features; by using them you also agree
            to Google’s terms and grant only the scopes you approve.
          </li>
        </ul>
      </section>

      <section>
        <h2>4. Acceptable use</h2>
        <p>You agree not to:</p>
        <ul>
          <li>Send spam, unlawful, deceptive, or harassing email.</li>
          <li>Violate anti-spam, privacy, advertising, or export laws.</li>
          <li>Upload malware or attempt to disrupt or reverse-engineer the Service.</li>
          <li>Use scraped or uploaded data in ways that infringe others’ rights.</li>
          <li>Share login credentials or access another customer’s workspace without permission.</li>
        </ul>
        <p>
          You are solely responsible for the content of emails you send and for obtaining any
          required consent from recipients.
        </p>
      </section>

      <section>
        <h2>5. Customer content</h2>
        <p>
          You retain ownership of leads, catalogues, prompts, templates, and messages you upload
          or create (“Customer Content”). You grant Starlight a limited licence to host, process,
          and display Customer Content solely to provide the Service. AI-generated drafts are
          provided as suggestions; you must review before sending.
        </p>
      </section>

      <section>
        <h2>6. Third-party services</h2>
        <p>
          The Service may integrate Azure OpenAI, Google, cloud hosting, and other providers.
          Their availability and policies are outside our control. Outages or policy changes at
          those providers may affect features such as generation or Gmail sending.
        </p>
      </section>

      <section>
        <h2>7. Intellectual property</h2>
        <p>
          Starlight and its licensors own the Service software, branding, and documentation.
          Except for Customer Content, no rights are granted other than a limited, non-exclusive
          licence to use the Service during your subscription or authorised access period.
        </p>
      </section>

      <section>
        <h2>8. Confidentiality</h2>
        <p>
          Each party will protect the other’s confidential information with reasonable care and
          use it only as needed to perform under these Terms.
        </p>
      </section>

      <section>
        <h2>9. Disclaimers</h2>
        <p>
          THE SERVICE IS PROVIDED “AS IS” AND “AS AVAILABLE”. TO THE MAXIMUM EXTENT PERMITTED BY
          LAW, STARLIGHT DISCLAIMS WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE,
          AND NON-INFRINGEMENT. WE DO NOT GUARANTEE deliverability, inbox placement, uninterrupted
          uptime, or that AI output is accurate or complete.
        </p>
      </section>

      <section>
        <h2>10. Limitation of liability</h2>
        <p>
          TO THE MAXIMUM EXTENT PERMITTED BY LAW, STARLIGHT WILL NOT BE LIABLE FOR INDIRECT,
          INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES, OR LOST PROFITS, REVENUE, OR
          DATA. OUR TOTAL LIABILITY FOR CLAIMS ARISING OUT OF THE SERVICE IS LIMITED TO THE AMOUNTS
          YOU PAID US FOR THE SERVICE IN THE THREE (3) MONTHS BEFORE THE CLAIM (OR INR 10,000 if
          you paid nothing).
        </p>
      </section>

      <section>
        <h2>11. Suspension and termination</h2>
        <p>
          We may suspend or terminate access if you breach these Terms, create security risk, or
          misuse the Service. You may stop using the Service at any time. Provisions that by nature
          should survive (including IP, disclaimers, and liability limits) will survive termination.
        </p>
      </section>

      <section>
        <h2>12. Changes to the Terms</h2>
        <p>
          We may update these Terms by posting a revised version with a new “Last updated” date.
          Material changes may be communicated in-product or by email when practical. Continued use
          after changes take effect constitutes acceptance.
        </p>
      </section>

      <section>
        <h2>13. Governing law</h2>
        <p>
          These Terms are governed by the laws of India, without regard to conflict-of-law rules.
          Courts in Pune, Maharashtra shall have exclusive jurisdiction, subject to mandatory
          consumer protections that may apply.
        </p>
      </section>

      <section>
        <h2>14. Contact</h2>
        <p>
          {COMPANY}<br />
          Email: <a href={`mailto:${CONTACT}`}>{CONTACT}</a><br />
          Website: <a href="https://www.starlightlinearled.com" target="_blank" rel="noreferrer">starlightlinearled.com</a>
        </p>
      </section>
    </LegalShell>
  )
}
