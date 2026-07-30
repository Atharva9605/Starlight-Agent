import { useEditor, EditorContent } from '@tiptap/react'
import StarterKit from '@tiptap/starter-kit'
import Link from '@tiptap/extension-link'
import Placeholder from '@tiptap/extension-placeholder'
import { useEffect } from 'react'

type Props = {
  html: string
  onChange: (html: string, plainText: string) => void
  placeholder?: string
}

export function EmailComposer({ html, onChange, placeholder = 'Write your reply…' }: Props) {
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

  const btn = (active: boolean) =>
    ({
      border: '1px solid var(--border)',
      background: active ? 'var(--accent-soft)' : 'white',
      color: active ? 'var(--blue-deep)' : 'var(--text)',
      borderRadius: 8,
      padding: '0.35rem 0.55rem',
      cursor: 'pointer',
      fontWeight: 700,
      fontSize: 13,
    }) as const

  return (
    <div className="composer">
      <div className="composer-toolbar row">
        <button type="button" style={btn(editor.isActive('bold'))} onClick={() => editor.chain().focus().toggleBold().run()}>
          B
        </button>
        <button type="button" style={btn(editor.isActive('italic'))} onClick={() => editor.chain().focus().toggleItalic().run()}>
          I
        </button>
        <button type="button" style={btn(editor.isActive('bulletList'))} onClick={() => editor.chain().focus().toggleBulletList().run()}>
          • List
        </button>
        <button type="button" style={btn(editor.isActive('orderedList'))} onClick={() => editor.chain().focus().toggleOrderedList().run()}>
          1. List
        </button>
        <button
          type="button"
          style={btn(editor.isActive('link'))}
          onClick={() => {
            const prev = editor.getAttributes('link').href as string | undefined
            const url = window.prompt('Link URL', prev || 'https://')
            if (url === null) return
            if (!url) {
              editor.chain().focus().extendMarkRange('link').unsetLink().run()
              return
            }
            editor.chain().focus().extendMarkRange('link').setLink({ href: url }).run()
          }}
        >
          Link
        </button>
      </div>
      <EditorContent editor={editor} className="composer-editor" />
    </div>
  )
}
