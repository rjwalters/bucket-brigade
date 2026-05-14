import { memo } from 'react';

// Research insight data is loosely typed in the underlying JSON config.
// We preserve the existing `any` shape used in the original page.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type RawInsight = any;

interface InsightGroupProps {
  title: string;
  titleClass: string;
  borderClass: string;
  bulletClass: string;
  implicationBg: string;
  implicationText: string;
  insights: RawInsight[];
}

function InsightGroup({
  title,
  titleClass,
  borderClass,
  bulletClass,
  implicationBg,
  implicationText,
  insights,
}: InsightGroupProps) {
  if (!insights || insights.length === 0) return null;

  return (
    <div className="mb-8">
      <h3 className={`text-xl font-bold mb-4 ${titleClass}`}>{title}</h3>
      <div className="space-y-6">
        {insights.map((insight: RawInsight, idx: number) => (
          <div
            key={idx}
            className={`bg-surface-secondary p-6 rounded-lg shadow border-l-4 ${borderClass}`}
          >
            <h4 className={`text-lg font-semibold mb-3 ${titleClass}`}>{insight.question}</h4>

            <div className="mb-4">
              <div className="text-sm font-semibold text-content-secondary uppercase mb-2">
                Finding
              </div>
              <p className="text-base text-content-primary">{insight.finding}</p>
            </div>

            {insight.evidence && insight.evidence.length > 0 && (
              <div className="mb-4">
                <div className="text-sm font-semibold text-content-secondary uppercase mb-2">
                  Evidence
                </div>
                <ul className="space-y-1">
                  {insight.evidence.map((evidence: string, evidx: number) => (
                    <li key={evidx} className="flex items-start text-sm">
                      <span className={`mr-2 mt-1 ${bulletClass}`}>•</span>
                      <span className="text-content-secondary">{evidence}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <div className={`${implicationBg} p-4 rounded`}>
              <div
                className={`text-sm font-semibold uppercase mb-1 ${implicationText}`}
              >
                Implication
              </div>
              <p className="text-sm text-content-primary italic">{insight.implication}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export interface ResearchInsightsSectionProps {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  config: any;
}

function ResearchInsightsSectionImpl({ config }: ResearchInsightsSectionProps) {
  const methodInsights = config?.method_insights;
  const hasMethodInsights = methodInsights && Object.keys(methodInsights).length > 0;
  const researchInsights = config?.research_insights;
  const hasResearchInsights = researchInsights && researchInsights.length > 0;
  const researchQuestions = config?.research_questions;
  const hasQuestions = researchQuestions && researchQuestions.length > 0;

  // 1. Structured method insights (preferred)
  if (hasMethodInsights) {
    return (
      <div className="mb-8">
        <h2 className="text-2xl font-bold mb-6 text-content-primary">Research Insights</h2>

        <InsightGroup
          title="Nash Equilibrium Analysis"
          titleClass="text-purple-700 dark:text-purple-300"
          borderClass="border-purple-500"
          bulletClass="text-purple-500"
          implicationBg="bg-purple-50 dark:bg-purple-900/20"
          implicationText="text-purple-600 dark:text-purple-300"
          insights={methodInsights.nash}
        />

        <InsightGroup
          title="Evolutionary Optimization"
          titleClass="text-green-700 dark:text-green-300"
          borderClass="border-green-500"
          bulletClass="text-green-500"
          implicationBg="bg-green-50 dark:bg-green-900/20"
          implicationText="text-green-600 dark:text-green-300"
          insights={methodInsights.evolution}
        />

        <InsightGroup
          title="Comparative Analysis"
          titleClass="text-blue-700 dark:text-blue-300"
          borderClass="border-blue-500"
          bulletClass="text-blue-500"
          implicationBg="bg-blue-50 dark:bg-blue-900/20"
          implicationText="text-blue-600 dark:text-blue-300"
          insights={methodInsights.comparative}
        />

        <InsightGroup
          title="PPO Training Analysis"
          titleClass="text-orange-700 dark:text-orange-300"
          borderClass="border-orange-500"
          bulletClass="text-orange-500"
          implicationBg="bg-orange-50 dark:bg-orange-900/20"
          implicationText="text-orange-600 dark:text-orange-300"
          insights={methodInsights.ppo}
        />
      </div>
    );
  }

  // 2. Legacy research insights
  if (hasResearchInsights) {
    return (
      <div className="mb-8">
        <h2 className="text-2xl font-bold mb-6 text-content-primary">Research Insights</h2>
        <div className="space-y-6">
          {researchInsights.map((insight: RawInsight, idx: number) => (
            <div
              key={idx}
              className="bg-surface-secondary p-6 rounded-lg shadow border-l-4 border-blue-500"
            >
              <h3 className="text-xl font-semibold mb-3 text-blue-700 dark:text-blue-300">
                {insight.question}
              </h3>

              <div className="mb-4">
                <div className="text-sm font-semibold text-content-secondary uppercase mb-2">
                  Finding
                </div>
                <p className="text-lg text-content-primary">{insight.finding}</p>
              </div>

              {insight.evidence && insight.evidence.length > 0 && (
                <div className="mb-4">
                  <div className="text-sm font-semibold text-content-secondary uppercase mb-2">
                    Evidence
                  </div>
                  <ul className="space-y-2">
                    {insight.evidence.map((evidence: string, evidx: number) => (
                      <li key={evidx} className="flex items-start">
                        <span className="text-green-500 mr-2 mt-1">•</span>
                        <span className="text-content-secondary">{evidence}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded">
                <div className="text-sm font-semibold text-blue-600 dark:text-blue-300 uppercase mb-1">
                  Implication
                </div>
                <p className="text-content-primary italic">{insight.implication}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // 3. Fall back to research questions
  if (hasQuestions) {
    return (
      <div className="mb-8">
        <h2 className="text-2xl font-bold mb-4 text-content-primary">Research Questions</h2>
        <div className="bg-surface-secondary p-6 rounded-lg shadow border border-outline-primary">
          <ul className="list-disc list-inside space-y-2">
            {researchQuestions.map((question: string, idx: number) => (
              <li key={idx} className="text-content-secondary">
                {question}
              </li>
            ))}
          </ul>
        </div>
      </div>
    );
  }

  return null;
}

export const ResearchInsightsSection = memo(ResearchInsightsSectionImpl);
