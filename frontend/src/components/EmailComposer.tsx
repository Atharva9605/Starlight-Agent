import { useEditor, EditorContent } from '@tiptap/react'
import StarterKit from '@tiptap/starter-kit'
import Link from '@tiptap/extension-link'
import Placeholder from '@tiptap/extension-placeholder'
import { useEffect, useState, type ReactNode } from 'react'
import {
  ArrowRedoRegular,
  ArrowUndoRegular,
  LinkRegular,
  TextBoldRegular,
  TextBulletListLtrRegular,
  TextItalicRegular,
  TextNumberListLtrRegular,
} from '@fluentui/react-icons'

type Props = {
  html: string
  onChange: (html: string, plainText: string) => void
  placeholder?: string
}

function ToolButton({
  label,
  active,
  onClick,
  children,
  disabled,
}: {
  label: string
  active?: boolean
  onClick: () => void
  children: ReactNode
  disabled?: boolean
}) {
  return (
    <button
      type="button"
      className={`btn subtle icon-only${active ? ' active' : ''}`}
      title={label}
      aria-label={label}
      aria-pressed={active}
      disabled={disabled}
      onMouseDown={(e) => e.preventDefault()}
      onClick={onClick}
    >
      {children}
    </button>
  )
}

export function EmailComposer({ html, onChange, placeholder = 'Write your reply…' }: Props) {
  const [linkOpen, setLinkOpen] = useState(false)
  const [linkUrl, setLinkUrl] = useState('')
  const editor = useEditor({
    extensions: [
      StarterKit,
      Link.configure({ openOnClick: false }),
      Placeholder.configure({ placeholder }),
    ],
    content: html || '',
    onUpdate: ({ editor: ed }) => {
      onChange(ed.getHTML(), ed.getText())
    },
  })

  useEffect(() => {
    if (!editor) return
    const current = editor.getHTML()
    const next = html || ''
    // Only reset when external content changes (e.g. new AI draft)
    if (next && next !== current) {
      editor.commands.setContent(next, { emitUpdate: false })
    }
  }, [html, editor])

  if (!editor) return null

  const openLink = () => {
    setLinkUrl((editor.getAttributes('link').href as string | undefined) || 'https://')
    setLinkOpen(true)
  }
  const applyLink = () => {
    const chain = editor.chain().focus().extendMarkRange('link')
    if (!linkUrl.trim() || linkUrl.trim() === 'https://') chain.unsetLink().run()
    else chain.setLink({ href: linkUrl.trim() }).run()
    setLinkOpen(false)
  }

  return (
    <div className="composer">
      <div className="composer-toolbar" role="toolbar" aria-label="Formatting">
        <ToolButton label="Bold (Ctrl+B)" active={editor.isActive('bold')} onClick={() => editor.chain().focus().toggleBold().run()}>
          <TextBoldRegular />
        </ToolButton>
        <ToolButton label="Italic (Ctrl+I)" active={editor.isActive('italic')} onClick={() => editor.chain().focus().toggleItalic().run()}>
          <TextItalicRegular />
        </ToolButton>
        <span className="command-bar-sep" />
        <ToolButton label="Bulleted list" active={editor.isActive('bulletList')} onClick={() => editor.chain().focus().toggleBulletList().run()}>
          <TextBulletListLtrRegular />
        </ToolButton>
        <ToolButton label="Numbered list" active={editor.isActive('orderedList')} onClick={() => editor.chain().focus().toggleOrderedList().run()}>
          <TextNumberListLtrRegular />
        </ToolButton>
        <ToolButton label="Insert link" active={editor.isActive('link') || linkOpen} onClick={openLink}>
          <LinkRegular />
        </ToolButton>
        <span className="command-bar-sep" />
        <ToolButton label="Undo (Ctrl+Z)" disabled={!editor.can().undo()} onClick={() => editor.chain().focus().undo().run()}>
          <ArrowUndoRegular />
        </ToolButton>
        <ToolButton label="Redo (Ctrl+Y)" disabled={!editor.can().redo()} onClick={() => editor.chain().focus().redo().run()}>
          <ArrowRedoRegular />
        </ToolButton>
      </div>
      {linkOpen ? (
        <div className="composer-link">
          <input
            className="input"
            autoFocus
            value={linkUrl}
            aria-label="Link address"
            placeholder="https://"
            onChange={(e) => setLinkUrl(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault()
                applyLink()
              }
              if (e.key === 'Escape') setLinkOpen(false)
            }}
          />
          <button type="button" className="btn" onClick={applyLink}>Apply</button>
          <button type="button" className="btn secondary" onClick={() => setLinkOpen(false)}>Cancel</button>
        </div>
      ) : null}
      <EditorContent editor={editor} className="composer-editor" />
    </div>
  )
}
