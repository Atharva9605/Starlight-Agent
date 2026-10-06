import { Link } from 'react-router-dom'
import { CompassNorthwestRegular, HomeRegular } from '@fluentui/react-icons'
import { EmptyState, useDocumentTitle } from '../components/ui'

export function NotFoundPage() {
  useDocumentTitle('Page not found')
  return (
    <div className="notfound">
      <EmptyState
        icon={<CompassNorthwestRegular />}
        title="We couldn't find that page"
        description="The link may be out of date, or the page may have moved. Use the navigation or search to find what you need."
        actions={
          <Link to="/" className="btn">
            <HomeRegular /> Go to Home
          </Link>
        }
      />
    </div>
  )
}
