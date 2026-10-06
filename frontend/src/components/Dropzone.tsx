import { useRef, useState } from 'react'
import { CloudArrowUpRegular } from '@fluentui/react-icons'
import { Spinner } from './ui'

type Props = {
  accept: string
  multiple?: boolean
  title: string
  hint: string
  busy?: boolean
  onFiles: (files: FileList) => void
}

export function Dropzone({ accept, multiple = false, title, hint, busy = false, onFiles }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [over, setOver] = useState(false)

  const handle = (files: FileList | null) => {
    if (files?.length && !busy) onFiles(files)
  }
  const browse = () => {
    if (!busy) inputRef.current?.click()
  }

  return (
    <div
      className={`dropzone${over ? ' over' : ''}${busy ? ' busy' : ''}`}
      onDragOver={(e) => {
        e.preventDefault()
        setOver(true)
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault()
        setOver(false)
        handle(e.dataTransfer.files)
      }}
      onClick={browse}
      role="button"
      tabIndex={0}
      aria-disabled={busy}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          browse()
        }
      }}
    >
      {busy ? (
        <Spinner size="lg" />
      ) : (
        <CloudArrowUpRegular className="dropzone-icon" />
      )}
      <strong>{busy ? 'Working…' : title}</strong>
      <span className="dropzone-hint">
        {busy ? 'You can keep using the app while this finishes.' : (
          <>
            or <span className="dropzone-browse">browse your files</span> · {hint}
          </>
        )}
      </span>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        multiple={multiple}
        hidden
        onChange={(e) => {
          handle(e.target.files)
          e.target.value = ''
        }}
      />
    </div>
  )
}
