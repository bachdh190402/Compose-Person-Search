import { useNavigate } from "react-router-dom";
import { Search, ArrowRight } from "lucide-react";
import PageTransition from "./layout/page-transition";
import Button from "./ui/button";

export default function HomePage() {
  const navigate = useNavigate();

  return (
    <PageTransition>
      <div className="min-h-[calc(100vh-4rem)]">
        {/* Hero */}
        <section className="bg-gradient-to-br from-white via-blue-50 to-slate-100 py-20 sm:py-28 lg:py-36">
          <div className="mx-auto max-w-4xl px-4 text-center">
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-text-primary">
              Composed Image&#8209;Text Person Search
            </h1>
            <p className="mt-4 text-lg sm:text-xl text-text-secondary max-w-2xl mx-auto">
              Search for pedestrians by combining a reference image with a free&#8209;form
              text description of the desired modifications.
            </p>

            <div className="mt-8 flex items-center justify-center">
              <Button
                size="lg"
                icon={<Search className="h-5 w-5" />}
                onClick={() => navigate("/compose-search")}
              >
                Start Searching
                <ArrowRight className="h-4 w-4 ml-1" />
              </Button>
            </div>
          </div>
        </section>

        {/* How it works */}
        <section className="mx-auto max-w-4xl px-4 py-16">
          <h2 className="text-2xl font-bold text-text-primary text-center mb-10">How It Works</h2>
          <div className="grid sm:grid-cols-3 gap-8 text-center">
            {[
              { step: "1", title: "Upload Image", desc: "Provide a reference pedestrian image as the visual query." },
              { step: "2", title: "Describe Changes", desc: "Write what should be different, e.g. \"wearing a red jacket\"." },
              { step: "3", title: "Get Results", desc: "The system retrieves the most similar persons from the gallery." },
            ].map((item) => (
              <div key={item.step} className="space-y-3">
                <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-primary text-white text-lg font-bold">
                  {item.step}
                </div>
                <h3 className="font-semibold text-text-primary">{item.title}</h3>
                <p className="text-sm text-text-secondary">{item.desc}</p>
              </div>
            ))}
          </div>
        </section>
      </div>
    </PageTransition>
  );
}
