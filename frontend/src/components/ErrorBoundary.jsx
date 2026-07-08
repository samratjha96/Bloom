import { Component } from 'react'

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { error: null }
  }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    console.error('[ErrorBoundary]', error, info.componentStack)
  }

  render() {
    if (this.state.error) {
      return (
        <div className="min-h-[100dvh] bg-stone-50 flex flex-col items-center justify-center gap-4 px-6">
          <p className="text-rose-500 text-sm font-medium">Something went wrong.</p>
          <p className="text-stone-400 text-xs max-w-sm text-center">{this.state.error.message}</p>
          <button
            onClick={() => { this.setState({ error: null }); window.history.back(); }}
            className="text-emerald-600 hover:text-emerald-700 text-sm font-medium transition-colors"
          >
            Go back
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
