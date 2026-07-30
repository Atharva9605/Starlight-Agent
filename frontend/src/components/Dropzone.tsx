import { useRef, useState } from 'react'

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
    if (files?.length) onFiles(files)
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
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click()
      }}
    >
      <div className="dropzone-icon">{busy ? '⏳' : '⬆'}</div>
      <strong>{busy ? 'Working…' : title}</strong>
      <span className="muted">{hint}</span>
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
