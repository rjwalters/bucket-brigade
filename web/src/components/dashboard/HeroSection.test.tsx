import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { HeroSection } from './HeroSection'

describe('HeroSection', () => {
  it('renders the project title', () => {
    render(<HeroSection />)
    expect(
      screen.getByRole('heading', { level: 1, name: /bucket brigade/i })
    ).toBeInTheDocument()
  })

  it('renders the tagline copy', () => {
    render(<HeroSection />)
    expect(
      screen.getByText(/watch cooperation emerge \(or fail\) in a frontier town/i)
    ).toBeInTheDocument()
  })
})
