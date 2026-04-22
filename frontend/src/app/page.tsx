"use client";

import { motion, useInView, useMotionValue, useSpring } from "framer-motion";
import Image from "next/image";
import { useEffect, useRef } from "react";

const trafficDots = [
  {
    className:
      "left-[10%] top-[24%] h-2 w-2 bg-cyan-300 shadow-[0_0_18px_rgba(103,232,249,0.9)]",
    duration: 5.5,
    delay: 0.2,
    x: [0, 140, 0],
    y: [0, -42, 0],
  },
  {
    className:
      "right-[18%] top-[18%] h-3 w-3 bg-sky-400 shadow-[0_0_22px_rgba(56,189,248,0.9)]",
    duration: 6.3,
    delay: 1.1,
    x: [0, -110, 0],
    y: [0, 56, 0],
  },
  {
    className:
      "left-[16%] bottom-[24%] h-2.5 w-2.5 bg-blue-300 shadow-[0_0_18px_rgba(147,197,253,0.95)]",
    duration: 7.1,
    delay: 0.7,
    x: [0, 120, 0],
    y: [0, 38, 0],
  },
  {
    className:
      "right-[14%] bottom-[20%] h-2 w-2 bg-cyan-200 shadow-[0_0_16px_rgba(165,243,252,0.95)]",
    duration: 5.8,
    delay: 1.8,
    x: [0, -135, 0],
    y: [0, -28, 0],
  },
  {
    className:
      "left-[48%] top-[14%] h-2 w-2 bg-blue-400 shadow-[0_0_16px_rgba(96,165,250,0.9)]",
    duration: 6.8,
    delay: 0.9,
    x: [0, 72, 0],
    y: [0, 68, 0],
  },
  {
    className:
      "left-[52%] bottom-[14%] h-2.5 w-2.5 bg-sky-300 shadow-[0_0_20px_rgba(125,211,252,0.95)]",
    duration: 7.4,
    delay: 1.4,
    x: [0, -86, 0],
    y: [0, -58, 0],
  },
];

const trafficLines = [
  "left-[8%] top-[30%] w-[28vw] max-w-[380px] rotate-6",
  "right-[10%] top-[36%] w-[24vw] max-w-[320px] -rotate-12",
  "left-[20%] bottom-[18%] w-[22vw] max-w-[280px] -rotate-[10deg]",
  "right-[18%] bottom-[24%] w-[26vw] max-w-[340px] rotate-[14deg]",
];

const fadeUp = {
  hidden: { opacity: 0, y: 32 },
  visible: { opacity: 1, y: 0 },
};

const howItWorksCards = [
  {
    title: "Local traffic sensing",
    description:
      "Extracts real-time traffic features such as queue length, flow, and signal states at each intersection.",
  },
  {
    title: "AI decision making",
    description:
      "Applies a shared reinforcement learning policy to dynamically select signal phases.",
  },
  {
    title: "Coordinated intersections",
    description:
      "Enables communication between intersections to improve overall network flow.",
  },
];

const whyThisMattersCards = [
  {
    title: "Fixed systems are rigid",
    description:
      "Traditional fixed-time traffic signals cannot adapt to real-time traffic variability.",
  },
  {
    title: "Traffic is dynamic",
    description:
      "Urban traffic patterns change constantly due to demand, events, and time-of-day variations.",
  },
  {
    title: "Scalability challenge",
    description:
      "Most intelligent systems struggle to scale across different road networks and configurations.",
  },
];

const differentiators = [
  {
    title: "Network-agnostic",
    description:
      "Designed to operate across different traffic networks without retraining.",
  },
  {
    title: "Shared policy",
    description:
      "Uses a single learned policy applied across all intersections.",
  },
  {
    title: "Coordinated intelligence",
    description:
      "Captures interactions between neighboring intersections for better global optimization.",
  },
];

const results = [
  {
    value: 50,
    prefix: "+",
    suffix: "%",
    title: "throughput improvement (RILSA benchmark)",
    description:
      "Measured against the RILSA benchmark under the evaluated simulation setting.",
  },
  {
    value: 0,
    prefix: "",
    suffix: "",
    title: "reduced average queue length",
    description:
      "Observed lower average queue length under the learned control policy.",
  },
  {
    value: 0,
    prefix: "",
    suffix: "",
    title: "tested across multiple traffic network structures",
    description:
      "Evaluated on different network layouts to assess generalization beyond a single topology.",
  },
];

const teamMembers = [
  {
    name: "Hasan Haidar",
    email: "hih17@mail.aub.edu",
    phone: "+961 70 614 923",
    image: "/images/hasan.jpg",
  },
  {
    name: "Mohamad Al Aalami",
    email: "maa399@mail.aub.edu",
    phone: "+961 70 144 745",
    image: "/images/aalami.jpg",
  },
  {
    name: "Mohammad Kassira",
    email: "msk58@mail.aub.edu",
    phone: "+961 70 495 274",
    image: "/images/kassira.jpg",
  },
];

type TeamMemberCardProps = {
  name: string;
  email: string;
  phone: string;
  image: string | null;
};

function TeamMemberCard({
  name,
  email,
  phone,
  image,
}: TeamMemberCardProps) {
  return (
    <div className="group rounded-3xl border border-white/10 bg-white/5 p-6 backdrop-blur-xl sm:p-8">
      <div className="flex flex-col gap-6 md:flex-row md:items-center">
        <div className="relative h-[260px] w-full overflow-hidden rounded-xl shadow-[0_0_32px_rgba(56,189,248,0.12)] md:h-[260px] md:w-[220px] md:shrink-0">
          {image ? (
            <Image
              src={image}
              alt={name}
              fill
              className="object-cover rounded-xl"
              sizes="(max-width: 768px) 100vw, 220px"
            />
          ) : (
            <div className="h-full w-full rounded-xl bg-[linear-gradient(135deg,rgba(255,255,255,0.08),rgba(56,189,248,0.12),rgba(37,99,235,0.18))]" />
          )}
        </div>

        <div className="min-w-0 space-y-3">
          <h3 className="text-2xl font-semibold text-white">{name}</h3>
          <p className="break-words text-slate-300">{email}</p>
          <p className="text-slate-300">{phone}</p>
        </div>
      </div>
    </div>
  );
}

function CountUp({
  value,
  prefix = "",
  suffix = "",
}: {
  value: number;
  prefix?: string;
  suffix?: string;
}) {
  const ref = useRef<HTMLSpanElement | null>(null);
  const isInView = useInView(ref, { once: true, margin: "-80px" });
  const motionValue = useMotionValue(0);
  const springValue = useSpring(motionValue, { damping: 20, stiffness: 90 });

  useEffect(() => {
    if (isInView) {
      motionValue.set(value);
    }
  }, [isInView, motionValue, value]);

  useEffect(() => {
    return springValue.on("change", (latest) => {
      if (ref.current) {
        ref.current.textContent = `${prefix}${Math.round(latest)}${suffix}`;
      }
    });
  }, [prefix, springValue, suffix]);

  return <span ref={ref}>{`${prefix}0${suffix}`}</span>;
}

export default function Home() {
  return (
    <main className="relative isolate overflow-hidden bg-[radial-gradient(circle_at_top,rgba(37,99,235,0.25),transparent_38%),linear-gradient(135deg,#02040a_0%,#08101f_42%,#0b1f45_100%)] text-white">
      <style jsx global>{`
        html {
          scroll-behavior: smooth;
        }
      `}</style>

      <div className="absolute inset-0 bg-[linear-gradient(120deg,rgba(255,255,255,0.05),transparent_30%,transparent_70%,rgba(56,189,248,0.07))]" />

      <motion.div
        className="absolute left-1/2 top-[-12rem] h-[28rem] w-[28rem] -translate-x-1/2 rounded-full bg-cyan-400/16 blur-3xl"
        animate={{
          scale: [1, 1.2, 1],
          x: ["-50%", "-44%", "-50%"],
          opacity: [0.3, 0.55, 0.3],
        }}
        transition={{ duration: 12, repeat: Infinity, ease: "easeInOut" }}
      />
      <motion.div
        className="absolute bottom-[18%] right-[-8rem] h-[26rem] w-[26rem] rounded-full bg-blue-500/16 blur-3xl"
        animate={{
          scale: [1, 1.16, 1],
          x: [0, -40, 0],
          y: [0, -30, 0],
          opacity: [0.22, 0.42, 0.22],
        }}
        transition={{ duration: 11, repeat: Infinity, ease: "easeInOut" }}
      />

      <div className="absolute inset-0 opacity-70">
        {trafficLines.map((line, index) => (
          <motion.div
            key={line}
            className={`absolute h-px overflow-hidden rounded-full bg-white/10 ${line}`}
            initial={{ opacity: 0.2 }}
            animate={{ opacity: [0.18, 0.35, 0.18] }}
            transition={{
              duration: 3.6 + index,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          >
            <motion.div
              className="h-full w-24 rounded-full bg-gradient-to-r from-transparent via-cyan-300 to-transparent blur-[1px]"
              animate={{ x: ["-20%", "130%"] }}
              transition={{
                duration: 4.8 + index * 0.7,
                repeat: Infinity,
                ease: "linear",
                delay: index * 0.6,
              }}
            />
          </motion.div>
        ))}

        {trafficDots.map((dot) => (
          <motion.span
            key={dot.className}
            className={`absolute rounded-full ${dot.className}`}
            animate={{
              x: dot.x,
              y: dot.y,
              opacity: [0.25, 0.95, 0.25],
              scale: [0.9, 1.25, 0.9],
            }}
            transition={{
              duration: dot.duration,
              delay: dot.delay,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />
        ))}
      </div>

      <header className="fixed inset-x-0 top-0 z-30 px-4 py-4 sm:px-6 lg:px-10">
        <div className="mx-auto flex max-w-7xl items-center justify-between rounded-full border border-white/10 bg-slate-950/30 px-5 py-3 backdrop-blur-2xl">
          {/* If the updated logo does not appear immediately, restart the Next.js dev server to clear cached assets. */}
          <a href="#home" className="flex items-center gap-3">
            <div className="relative w-12 h-12 rounded-full overflow-hidden">
              <Image
                src="/images/logo.png"
                alt="LightMind Logo"
                fill
                className="object-cover"
              />
            </div>
            <span className="text-white font-semibold tracking-[0.24em]">
              LightMind
            </span>
          </a>
          <nav className="hidden items-center gap-2 text-sm text-slate-200 md:flex">
            {[
              ["Home", "#home"],
              ["Why It Matters", "#why-it-matters"],
              ["How It Works", "#how-it-works"],
              ["Approach", "#approach"],
              ["Results", "#results"],
              ["Contact", "#contact"],
            ].map(([item, href]) => (
              <motion.a
                key={item}
                href={href}
                whileHover={{ y: -1 }}
                className="rounded-full px-4 py-2 transition duration-300 hover:bg-white/8 hover:text-cyan-200"
              >
                {item}
              </motion.a>
            ))}
          </nav>
        </div>
      </header>

      <section
        id="home"
        className="relative z-10 flex min-h-screen items-center justify-center px-6 py-24 sm:px-10 lg:px-16"
      >
        <div className="mx-auto flex w-full max-w-6xl items-center justify-center">
          <div className="mx-auto flex max-w-4xl flex-col items-center text-center">
            <motion.div
              className="mb-6 inline-flex items-center gap-3 rounded-full border border-white/12 bg-white/6 px-4 py-2 text-xs font-medium uppercase tracking-[0.28em] text-cyan-100/80 backdrop-blur-xl"
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
            >
              <span className="h-2 w-2 rounded-full bg-cyan-300 shadow-[0_0_14px_rgba(103,232,249,0.95)]" />
              Adaptive Urban Intelligence
            </motion.div>

            <motion.h1
              className="max-w-5xl text-balance text-5xl font-semibold tracking-[-0.06em] text-white sm:text-6xl lg:text-8xl"
              variants={fadeUp}
              initial="hidden"
              animate="visible"
              transition={{ duration: 0.95, ease: [0.22, 1, 0.36, 1] }}
            >
              AI Traffic Control That Adapts to Any City
            </motion.h1>

            <motion.p
              className="mt-6 max-w-2xl text-pretty text-base leading-8 text-slate-300 sm:text-lg"
              variants={fadeUp}
              initial="hidden"
              animate="visible"
              transition={{
                duration: 0.9,
                delay: 0.18,
                ease: [0.22, 1, 0.36, 1],
              }}
            >
              A generalizable reinforcement learning system designed to operate
              across different traffic networks without retraining.
            </motion.p>

            <motion.div
              className="mt-10"
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{
                duration: 0.85,
                delay: 0.32,
                ease: [0.22, 1, 0.36, 1],
              }}
            >
              <motion.a
                href="#how-it-works"
                whileHover={{ scale: 1.04 }}
                whileTap={{ scale: 0.98 }}
                transition={{ type: "spring", stiffness: 260, damping: 18 }}
                className="group relative inline-flex items-center justify-center rounded-full border border-cyan-300/30 bg-gradient-to-r from-cyan-300 via-sky-400 to-blue-500 px-7 py-3.5 text-sm font-semibold text-slate-950 shadow-[0_0_0_rgba(56,189,248,0)] duration-300 hover:shadow-[0_0_36px_rgba(56,189,248,0.42)]"
              >
                <span className="absolute inset-0 -z-10 rounded-full bg-cyan-300/25 opacity-0 blur-xl transition-opacity duration-300 group-hover:opacity-100" />
                Get Started
              </motion.a>
            </motion.div>

            <motion.div
              className="mt-14 grid w-full max-w-3xl grid-cols-1 gap-4 sm:grid-cols-3"
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{
                duration: 0.9,
                delay: 0.45,
                ease: [0.22, 1, 0.36, 1],
              }}
            >
              {[
                ["Instant rollout", "Deploy signal intelligence in existing intersections."],
                ["City-scale learning", "Coordinate corridors with real-time adaptive policies."],
                ["Always-on insight", "Surface congestion patterns before they cascade."],
              ].map(([title, description]) => (
                <div
                  key={title}
                  className="rounded-3xl border border-white/10 bg-white/6 p-5 text-left backdrop-blur-xl"
                >
                  <p className="text-sm font-semibold text-white">{title}</p>
                  <p className="mt-2 text-sm leading-6 text-slate-300">
                    {description}
                  </p>
                </div>
              ))}
            </motion.div>

            <motion.p
              className="mt-8 max-w-2xl text-sm leading-7 text-slate-400"
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{
                duration: 0.85,
                delay: 0.56,
                ease: [0.22, 1, 0.36, 1],
              }}
            >
              Evaluated using SUMO traffic simulation on multiple network
              topologies.
            </motion.p>
          </div>
        </div>
      </section>

      <section
        id="why-it-matters"
        className="relative z-10 px-6 py-24 sm:px-10 lg:px-16"
      >
        <motion.div
          className="mx-auto max-w-6xl"
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.2 }}
          variants={fadeUp}
          transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
        >
          <div className="mx-auto max-w-2xl text-center">
            <p className="text-sm font-medium uppercase tracking-[0.32em] text-cyan-200/75">
              Why This Matters
            </p>
            <h2 className="mt-4 text-4xl font-semibold tracking-[-0.05em] text-white sm:text-5xl">
              Why Traffic Control Needs to Change
            </h2>
          </div>

          <div className="mt-14 grid gap-6 md:grid-cols-3">
            {whyThisMattersCards.map((card, index) => (
              <motion.div
                key={card.title}
                className="group rounded-[2rem] border border-white/10 bg-white/6 p-7 backdrop-blur-2xl"
                initial={{ opacity: 0, y: 28 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, amount: 0.3 }}
                transition={{
                  duration: 0.75,
                  delay: 0.12 * index,
                  ease: [0.22, 1, 0.36, 1],
                }}
                whileHover={{ y: -8 }}
              >
                <h3 className="text-2xl font-semibold tracking-[-0.03em] text-white">
                  {card.title}
                </h3>
                <p className="mt-4 text-base leading-7 text-slate-300">
                  {card.description}
                </p>
                <div className="mt-8 h-px w-full bg-gradient-to-r from-cyan-300/0 via-cyan-300/50 to-cyan-300/0 opacity-50 transition duration-300 group-hover:opacity-100" />
              </motion.div>
            ))}
          </div>
        </motion.div>
      </section>

      <section
        id="how-it-works"
        className="relative z-10 px-6 py-24 sm:px-10 lg:px-16"
      >
        <motion.div
          className="mx-auto max-w-6xl"
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.2 }}
          variants={fadeUp}
          transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
        >
          <div className="mx-auto max-w-2xl text-center">
            <p className="text-sm font-medium uppercase tracking-[0.32em] text-cyan-200/75">
              How It Works
            </p>
            <h2 className="mt-4 text-4xl font-semibold tracking-[-0.05em] text-white sm:text-5xl">
              From street-level data to city-wide signal coordination
            </h2>
          </div>

          <div className="mt-14 grid gap-6 md:grid-cols-3">
            {howItWorksCards.map((card, index) => (
              <motion.div
                key={card.title}
                className="group rounded-[2rem] border border-white/10 bg-white/6 p-7 backdrop-blur-2xl"
                initial={{ opacity: 0, y: 28 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, amount: 0.3 }}
                transition={{
                  duration: 0.75,
                  delay: 0.12 * index,
                  ease: [0.22, 1, 0.36, 1],
                }}
                whileHover={{ y: -8 }}
              >
                <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-cyan-300/20 bg-cyan-300/10 text-sm font-semibold text-cyan-200 shadow-[0_0_24px_rgba(56,189,248,0.15)]">
                  0{index + 1}
                </div>
                <h3 className="mt-6 text-2xl font-semibold tracking-[-0.03em] text-white">
                  {card.title}
                </h3>
                <p className="mt-4 text-base leading-7 text-slate-300">
                  {card.description}
                </p>
                <div className="mt-8 h-px w-full bg-gradient-to-r from-cyan-300/0 via-cyan-300/50 to-cyan-300/0 opacity-50 transition duration-300 group-hover:opacity-100" />
              </motion.div>
            ))}
          </div>
        </motion.div>
      </section>

      <section
        id="approach"
        className="relative z-10 px-6 py-24 sm:px-10 lg:px-16"
      >
        <motion.div
          className="mx-auto max-w-6xl"
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.2 }}
          variants={fadeUp}
          transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
        >
          <div className="mx-auto max-w-2xl text-center">
            <p className="text-sm font-medium uppercase tracking-[0.32em] text-cyan-200/75">
              Differentiation
            </p>
            <h2 className="mt-4 text-4xl font-semibold tracking-[-0.05em] text-white sm:text-5xl">
              What Makes This Approach Different
            </h2>
          </div>

          <div className="mt-14 grid gap-6 md:grid-cols-3">
            {differentiators.map((card, index) => (
              <motion.div
                key={card.title}
                className="group rounded-[2rem] border border-white/10 bg-white/6 p-7 backdrop-blur-2xl"
                initial={{ opacity: 0, y: 28 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, amount: 0.3 }}
                transition={{
                  duration: 0.75,
                  delay: 0.12 * index,
                  ease: [0.22, 1, 0.36, 1],
                }}
                whileHover={{ y: -8 }}
              >
                <h3 className="text-2xl font-semibold tracking-[-0.03em] text-white">
                  {card.title}
                </h3>
                <p className="mt-4 text-base leading-7 text-slate-300">
                  {card.description}
                </p>
                <div className="mt-8 h-px w-full bg-gradient-to-r from-cyan-300/0 via-cyan-300/50 to-cyan-300/0 opacity-50 transition duration-300 group-hover:opacity-100" />
              </motion.div>
            ))}
          </div>
        </motion.div>
      </section>

      <section id="results" className="relative z-10 px-6 py-24 sm:px-10 lg:px-16">
        <motion.div
          className="mx-auto max-w-6xl rounded-[2.5rem] border border-white/10 bg-slate-950/35 p-8 backdrop-blur-2xl sm:p-10 lg:p-14"
          initial={{ opacity: 0, y: 32 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.2 }}
          transition={{ duration: 0.85, ease: [0.22, 1, 0.36, 1] }}
        >
          <div className="grid gap-12 lg:grid-cols-[0.9fr_1.1fr] lg:items-center">
            <div>
              <p className="text-sm font-medium uppercase tracking-[0.32em] text-cyan-200/75">
                Results
              </p>
              <h2 className="mt-4 text-4xl font-semibold tracking-[-0.05em] text-white sm:text-5xl">
                Proven impact where timing decisions matter most
              </h2>
              <p className="mt-6 max-w-xl text-base leading-8 text-slate-300">
                The system is evaluated in SUMO traffic simulations across
                different network settings to assess performance and
                generalization.
              </p>
            </div>

            <div className="grid items-stretch gap-5 sm:grid-cols-3">
              {results.map((result, index) => (
                <motion.div
                  key={result.title}
                  className="min-w-0 overflow-hidden rounded-[2rem] border border-white/10 bg-white/6 p-6 backdrop-blur-xl"
                  initial={{ opacity: 0, y: 24 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, amount: 0.4 }}
                  transition={{
                    duration: 0.7,
                    delay: 0.12 * index,
                    ease: [0.22, 1, 0.36, 1],
                  }}
                >
                  <p className="break-words text-balance text-2xl leading-tight font-semibold tracking-[-0.05em] text-white sm:text-3xl">
                    {result.value > 0 ? (
                      <CountUp
                        value={result.value}
                        prefix={result.prefix}
                        suffix={result.suffix}
                      />
                    ) : (
                      result.title
                    )}
                  </p>
                  {result.value > 0 ? (
                    <p className="mt-4 text-sm font-semibold uppercase tracking-[0.22em] text-cyan-200/80">
                      {result.title}
                    </p>
                  ) : null}
                  <p className="mt-3 text-sm leading-6 text-slate-300">
                    {result.description}
                  </p>
                </motion.div>
              ))}
            </div>
          </div>

          <p className="mt-8 text-sm leading-7 text-slate-400">
            Evaluated using SUMO traffic simulation on multiple network
            topologies.
          </p>
        </motion.div>
      </section>

      <section
        id="contact"
        className="relative z-10 px-6 py-24 sm:px-10 lg:px-16"
      >
        <motion.div
          className="mx-auto max-w-6xl"
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.2 }}
          variants={fadeUp}
          transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
        >
          <div className="mx-auto max-w-2xl text-center">
            <p className="text-sm font-medium uppercase tracking-[0.32em] text-cyan-200/75">
              Contact
            </p>
            <h2 className="mt-4 text-4xl font-semibold tracking-[-0.05em] text-white sm:text-5xl">
              Contact Us
            </h2>
            <p className="mt-4 text-base leading-8 text-slate-300">
              Meet the team behind LightMind
            </p>
          </div>

          <div className="mt-14 grid gap-6">
            {teamMembers.map((member, index) => (
              <motion.div
                key={member.email}
                initial={{ opacity: 0, y: 28 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, amount: 0.25 }}
                transition={{
                  duration: 0.75,
                  delay: 0.12 * index,
                  ease: [0.22, 1, 0.36, 1],
                }}
                whileHover={{ y: -6 }}
              >
                <TeamMemberCard {...member} />
              </motion.div>
            ))}
          </div>
        </motion.div>
      </section>
    </main>
  );
}
