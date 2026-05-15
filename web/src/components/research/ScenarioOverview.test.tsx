import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { ScenarioOverview } from './ScenarioOverview'

const baseConfig = {
  description: 'Trivial Cooperation',
  story: 'A simple test scenario for verifying coordination.',
  parameters: {
    beta: 0.3,
    kappa: 1.2,
    A: 5,
    L: 3,
    c: 0.1,
    rho_ignite: 0.4,
    N_min: 10,
    p_spark: 0.05,
    N_spark: 2,
    num_agents: 4,
  },
}

describe('ScenarioOverview', () => {
  it('renders scenario description and story', () => {
    render(<ScenarioOverview config={baseConfig} />)
    expect(
      screen.getByRole('heading', { name: /trivial cooperation/i })
    ).toBeInTheDocument()
    expect(
      screen.getByText(/simple test scenario for verifying coordination/i)
    ).toBeInTheDocument()
  })

  it('renders parameter values from config', () => {
    render(<ScenarioOverview config={baseConfig} />)
    // num_agents is a unique value (4) — check label + value association
    expect(screen.getByText('Agents')).toBeInTheDocument()
    expect(screen.getByText('Fire Spread (β)')).toBeInTheDocument()
    expect(screen.getByText('0.3')).toBeInTheDocument() // beta
  })

  it('renders nothing when config is null', () => {
    const { container } = render(<ScenarioOverview config={null} />)
    expect(container).toBeEmptyDOMElement()
  })
})
