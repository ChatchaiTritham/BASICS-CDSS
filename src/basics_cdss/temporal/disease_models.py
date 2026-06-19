"""Physiological disease progression models for digital twin simulation.

This module implements disease-specific progression models based on
differential equations and clinical knowledge. Models simulate how
patient vital signs and lab values evolve over time.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class DiseaseModel(ABC):
    """Abstract base class for disease progression models.

    All disease models must implement the evolve() method which computes
    the next patient state given current state and interventions.
    """

    @abstractmethod
    def evolve(
        self,
        current_state: Dict[str, Any],
        dt: float,
        interventions: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, Any]:
        """Compute next patient state.

        Args:
            current_state: Current patient features
            dt: Time step in hours
            interventions: Applied interventions (e.g., medications, fluids)
            rng: Random number generator for stochastic evolution

        Returns:
            Next patient state after dt hours
        """
        pass


class SepsisModel(DiseaseModel):
    """Simplified sepsis progression model.

    Based on SIRS (Systemic Inflammatory Response Syndrome) criteria
    and compartmental infection dynamics.

    Simulated variables:
        - Temperature (°C)
        - Heart rate (bpm)
        - Respiratory rate (breaths/min)
        - White blood cell count (cells/μL)
        - Blood pressure (systolic mmHg)
        - Lactate (mmol/L)

    Interventions:
        - antibiotic: Reduces infection severity
        - fluid_bolus: Increases blood pressure, dilutes lactate
        - vasopressor: Increases blood pressure

    References:
        Singer et al. (2016). The Third International Consensus Definitions
        for Sepsis and Septic Shock (Sepsis-3). JAMA.
    """

    def __init__(
        self,
        infection_growth_rate: float = 0.15,
        infection_decay_rate: float = 0.3,
        temperature_sensitivity: float = 0.5,
        hemodynamic_sensitivity: float = 0.4,
        noise_std: float = 0.05,
        k_damage: float = 1.0,
        theta_damage: float = 0.2,
    ):
        """Initialize sepsis model parameters.

        Args:
            infection_growth_rate: Rate of infection worsening (per hour)
            infection_decay_rate: Rate of recovery with antibiotics
            temperature_sensitivity: How much temperature responds to infection
            hemodynamic_sensitivity: How much BP/HR respond to infection
            noise_std: Standard deviation of measurement noise
            k_damage: Gain on cumulative organ-damage accrual (per hour). See the
                organ-damage state D below.
            theta_damage: Pro-inflammatory severity threshold above which
                irreversible organ damage accrues (dimensionless, on the
                0-1 infection-severity scale).

        Organ-damage state (manuscript supplementary A.1):
            Sepsis causes irreversible end-organ injury that accumulates while
            the host is in a sustained pro-inflammatory state. We model a latent,
            monotone non-decreasing damage state D with

                dD/dt = k_damage * max(I_pro - theta_damage, 0)

            where I_pro is the (post-update) infection severity. Antibiotics lower
            I_pro and therefore halt FURTHER accrual, but D can NEVER decrease:
            already-injured tissue is not recovered on the simulation horizon.
            This is the mechanism that makes a delayed antibiotic worse than an
            early one -- a later intervention leaves a longer high-severity window
            and therefore a larger terminal D. Mechanism grounded in the
            time-dependent mortality of septic shock (Kumar et al., 2006, Crit
            Care Med: each hour of effective-antimicrobial delay is associated
            with measurable additional mortality).
        """
        self.infection_growth_rate = infection_growth_rate
        self.infection_decay_rate = infection_decay_rate
        self.temperature_sensitivity = temperature_sensitivity
        self.hemodynamic_sensitivity = hemodynamic_sensitivity
        self.noise_std = noise_std
        self.k_damage = k_damage
        self.theta_damage = theta_damage

    def evolve(
        self,
        current_state: Dict[str, Any],
        dt: float,
        interventions: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, Any]:
        """Evolve sepsis patient state."""
        if rng is None:
            rng = np.random.RandomState()

        # Extract current vitals (with defaults)
        temp = current_state.get('temperature', 37.0)
        hr = current_state.get('heart_rate', 80)
        rr = current_state.get('respiratory_rate', 16)
        wbc = current_state.get('white_blood_cell_count', 8000)
        bp_sys = current_state.get('blood_pressure_sys', 120)
        lactate = current_state.get('lactate', 1.0)

        # Cumulative irreversible organ-damage state D (carried across steps).
        # Initialised to 0 at t=0; monotone non-decreasing thereafter.
        organ_damage = current_state.get('_organ_damage', 0.0)

        # Estimate infection severity from current state
        # Higher temp, HR, lactate → higher severity
        infection_severity = (
            0.3 * max(0, (temp - 37.0) / 2.0)
            + 0.3 * max(0, (hr - 80) / 40.0)
            + 0.4 * max(0, (lactate - 1.0) / 3.0)
        )
        infection_severity = np.clip(infection_severity, 0, 1)

        # Intervention effects
        antibiotic_effect = 0.0
        fluid_effect = 0.0
        vasopressor_effect = 0.0

        if interventions:
            if interventions.get('antibiotic', False):
                antibiotic_effect = self.infection_decay_rate
            if interventions.get('fluid_bolus', 0) > 0:
                # Fluid bolus in mL
                fluid_effect = interventions['fluid_bolus'] / 1000.0  # normalize
            if interventions.get('vasopressor', False):
                vasopressor_effect = 0.5

        # Update infection severity
        d_infection = (
            self.infection_growth_rate * infection_severity * (1 - infection_severity)
            - antibiotic_effect * infection_severity
        ) * dt
        infection_severity = np.clip(infection_severity + d_infection, 0, 1)

        # Accrue irreversible organ damage from the (post-update) severity.
        # dD = k_damage * max(I_pro - theta_damage, 0) * dt, clamped >= 0 so D is
        # strictly monotone non-decreasing. Antibiotics act only by lowering
        # infection_severity above (halting accrual); they cannot reduce D.
        d_damage = self.k_damage * max(infection_severity - self.theta_damage, 0.0) * dt
        organ_damage = organ_damage + max(d_damage, 0.0)

        # Temperature dynamics
        # Fever develops with infection, resolves with antibiotics
        temp_target = 37.0 + 2.5 * infection_severity
        d_temp = (temp_target - temp) * self.temperature_sensitivity * dt
        temp_new = temp + d_temp + rng.normal(0, self.noise_std * dt)
        temp_new = np.clip(temp_new, 35.0, 41.0)

        # Heart rate dynamics
        # Tachycardia with infection and fever
        hr_target = 70 + 50 * infection_severity + 10 * (temp_new - 37.0)
        d_hr = (hr_target - hr) * self.hemodynamic_sensitivity * dt
        hr_new = hr + d_hr + rng.normal(0, 5 * dt)
        hr_new = np.clip(hr_new, 50, 180)

        # Respiratory rate
        rr_target = 14 + 12 * infection_severity
        d_rr = (rr_target - rr) * 0.3 * dt
        rr_new = rr + d_rr + rng.normal(0, 2 * dt)
        rr_new = np.clip(rr_new, 8, 40)

        # White blood cell count
        # Leukocytosis with infection
        wbc_target = 8000 + 15000 * infection_severity
        d_wbc = (wbc_target - wbc) * 0.2 * dt
        wbc_new = wbc + d_wbc + rng.normal(0, 500 * dt)
        wbc_new = np.clip(wbc_new, 2000, 30000)

        # Blood pressure (systolic)
        # Hypotension with severe sepsis, improved by fluids/vasopressors
        bp_target = (
            120 - 40 * infection_severity + 20 * fluid_effect + 30 * vasopressor_effect
        )
        d_bp = (bp_target - bp_sys) * 0.5 * dt
        bp_new = bp_sys + d_bp + rng.normal(0, 3 * dt)
        bp_new = np.clip(bp_new, 60, 180)

        # Lactate
        # Elevated with poor perfusion (low BP), cleared with fluids
        lactate_target = 1.0 + 3.0 * infection_severity - 1.0 * fluid_effect
        d_lactate = (lactate_target - lactate) * 0.3 * dt
        lactate_new = lactate + d_lactate + rng.normal(0, 0.1 * dt)
        lactate_new = np.clip(lactate_new, 0.5, 10.0)

        # Build next state (preserve non-modeled features)
        next_state = current_state.copy()
        next_state.update(
            {
                'temperature': float(temp_new),
                'heart_rate': float(hr_new),
                'respiratory_rate': float(rr_new),
                'white_blood_cell_count': float(wbc_new),
                'blood_pressure_sys': float(bp_new),
                'lactate': float(lactate_new),
                # Internal state (not observable)
                '_infection_severity': float(infection_severity),
                # Cumulative irreversible organ damage (latent outcome driver).
                '_organ_damage': float(organ_damage),
                'organ_damage': float(organ_damage),
            }
        )

        return next_state


class RespiratoryDistressModel(DiseaseModel):
    """Acute Respiratory Distress Syndrome (ARDS) progression model.

    Simulated variables:
        - Oxygen saturation (SpO2 %)
        - Respiratory rate (breaths/min)
        - PaO2/FiO2 ratio
        - Lung compliance
        - Heart rate

    Interventions:
        - oxygen_flow: Supplemental oxygen (L/min)
        - peep: Positive end-expiratory pressure (cmH2O)
        - prone_positioning: Improves oxygenation
    """

    def __init__(self, noise_std: float = 0.03, k_baro: float = 0.3):
        """Initialize ARDS model.

        Args:
            noise_std: Standard deviation of measurement noise.
            k_baro: Small gain on cumulative ventilator-associated lung injury.
                Compared with the sepsis organ-damage gain this is deliberately
                smaller: the dominant timing benefit in ARDS is the one-shot
                effect of instituting lung-protective ventilation, not a steep
                per-hour irreversible-accrual curve (ARDS Network, 2000, NEJM,
                low-tidal-volume ventilation). A small monotone barotrauma term
                captures the residual time dependence.
        """
        self.noise_std = noise_std
        self.k_baro = k_baro

    def evolve(
        self,
        current_state: Dict[str, Any],
        dt: float,
        interventions: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, Any]:
        """Evolve respiratory distress patient state."""
        if rng is None:
            rng = np.random.RandomState()

        # Cumulative (small) irreversible lung-injury state carried across steps.
        lung_damage = current_state.get('_lung_damage', 0.0)

        # Extract current state
        spo2 = current_state.get('oxygen_saturation', 98.0)
        rr = current_state.get('respiratory_rate', 16)
        pf_ratio = current_state.get('pf_ratio', 400)  # PaO2/FiO2
        hr = current_state.get('heart_rate', 80)

        # Estimate lung injury severity
        lung_injury = np.clip((400 - pf_ratio) / 300.0, 0, 1)

        # Small monotone cumulative barotrauma/lung-injury accrual.
        d_lung = self.k_baro * lung_injury * dt
        lung_damage = lung_damage + max(d_lung, 0.0)

        # Intervention effects
        oxygen_effect = 0.0
        peep_effect = 0.0
        prone_effect = 0.0

        if interventions:
            oxygen_flow = interventions.get('oxygen_flow', 0)
            oxygen_effect = min(oxygen_flow / 15.0, 1.0)  # Max at 15L/min

            peep = interventions.get('peep', 0)
            peep_effect = min(peep / 15.0, 0.5)  # Max benefit at 15 cmH2O

            if interventions.get('prone_positioning', False):
                prone_effect = 0.3

        # SpO2 dynamics
        spo2_target = (
            98 - 12 * lung_injury + 8 * oxygen_effect + 5 * (peep_effect + prone_effect)
        )
        d_spo2 = (spo2_target - spo2) * 0.4 * dt
        spo2_new = spo2 + d_spo2 + rng.normal(0, self.noise_std * 100 * dt)
        spo2_new = np.clip(spo2_new, 70, 100)

        # Respiratory rate (tachypnea with hypoxia)
        rr_target = 14 + 20 * lung_injury - 6 * oxygen_effect
        d_rr = (rr_target - rr) * 0.3 * dt
        rr_new = rr + d_rr + rng.normal(0, 2 * dt)
        rr_new = np.clip(rr_new, 8, 45)

        # PF ratio
        pf_target = (
            400 - 250 * lung_injury + 100 * (oxygen_effect + peep_effect + prone_effect)
        )
        d_pf = (pf_target - pf_ratio) * 0.2 * dt
        pf_new = pf_ratio + d_pf + rng.normal(0, 10 * dt)
        pf_new = np.clip(pf_new, 100, 500)

        # Heart rate (compensatory tachycardia)
        hr_target = 75 + 30 * lung_injury
        d_hr = (hr_target - hr) * 0.3 * dt
        hr_new = hr + d_hr + rng.normal(0, 3 * dt)
        hr_new = np.clip(hr_new, 50, 150)

        # Update state
        next_state = current_state.copy()
        next_state.update(
            {
                'oxygen_saturation': float(spo2_new),
                'respiratory_rate': float(rr_new),
                'pf_ratio': float(pf_new),
                'heart_rate': float(hr_new),
                '_lung_injury': float(lung_injury),
                '_lung_damage': float(lung_damage),
                'lung_damage': float(lung_damage),
            }
        )

        return next_state


class CardiacEventModel(DiseaseModel):
    """Acute cardiac event (MI/ACS) progression model.

    Simulated variables:
        - Heart rate (bpm)
        - Blood pressure (systolic/diastolic mmHg)
        - Troponin (ng/mL)
        - ST segment elevation (mm)
        - Chest pain score (0-10)

    Interventions:
        - aspirin: Antiplatelet
        - nitrate: Vasodilator
        - beta_blocker: Reduces heart rate and BP
        - pci: Percutaneous coronary intervention
    """

    def __init__(
        self,
        noise_std: float = 0.04,
        k_salvage: float = 1.0,
        salvage_lambda: float = 0.12,
        ischemia_progression: float = 0.04,
    ):
        """Initialize cardiac event model.

        Args:
            noise_std: Standard deviation of measurement noise.
            k_salvage: Base gain on irreversible myocardial-infarct accrual.
            salvage_lambda: Exponential salvage-decay rate (per hour). Myocardial
                salvage achievable by reperfusion decays roughly exponentially
                with time from symptom onset; we use lambda ~= 0.12/h (Terkelsen
                et al., 2010, JAMA -- system-delay / time-to-reperfusion vs
                mortality). See the infarct state below.
            ischemia_progression: Slow per-hour natural-history drift of latent
                ischemia toward full occlusion in an UNREPERFUSED vessel (per
                hour, dimensionless on the 0-1 ischemia scale). Kept small so an
                untreated STEMI worsens gradually rather than instantaneously.

        Latent-ischemia state (causal driver -- see ``evolve``):
            Ischemia is treated as a *latent state* carried across steps, seeded
            at t=0 from the presenting ST-elevation / troponin and thereafter
            evolving by its own natural history (slow worsening if unreperfused)
            and by reperfusion (PCI/nitrate lowering it). Troponin and ST are
            DOWNSTREAM readouts of this latent ischemia; they are computed FROM
            ischemia but never fed back into it. This removes the original
            self-amplifying loop (ischemia <- troponin/ST <- ischemia) that drove
            every cardiac twin -- regardless of presentation -- to full occlusion
            within a few hours, collapsing the cohort to a single mortality class.

        Cumulative infarct state (analogue of the sepsis organ-damage state):
            Acute coronary occlusion produces irreversible myocardium loss that
            accumulates while ischemia persists. The accrual rate is weighted by
            an exponentially decaying salvage window e^{-lambda * t}: tissue at
            risk early can still be salvaged (so its loss-rate weight is high and
            reperfusion timing matters most early), whereas after the salvage
            window has closed little additional myocardium remains to lose. We
            model a monotone non-decreasing infarct state D_inf with

                dD_inf/dt = k_salvage * ischemia * e^{-salvage_lambda * t}

            evaluated at the current trajectory time t. Reperfusion (PCI) lowers
            ischemia and so halts further accrual but cannot recover lost
            myocardium (D_inf never decreases).
        """
        self.noise_std = noise_std
        self.k_salvage = k_salvage
        self.salvage_lambda = salvage_lambda
        self.ischemia_progression = ischemia_progression

    def evolve(
        self,
        current_state: Dict[str, Any],
        dt: float,
        interventions: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, Any]:
        """Evolve cardiac patient state."""
        if rng is None:
            rng = np.random.RandomState()

        # Cumulative irreversible infarct state and elapsed time (carried across
        # steps). Initialised to 0 at t=0.
        infarct = current_state.get('_infarct', 0.0)
        elapsed = current_state.get('_elapsed_hours', 0.0)

        # Extract current state
        hr = current_state.get('heart_rate', 75)
        bp_sys = current_state.get('blood_pressure_sys', 130)
        bp_dia = current_state.get('blood_pressure_dia', 80)
        troponin = current_state.get('troponin', 0.01)
        st_elevation = current_state.get('st_elevation', 0.0)
        chest_pain = current_state.get('chest_pain_score', 0)

        # Latent ischemia as a CARRIED state (causal driver), not recomputed from
        # its own downstream markers. At t=0 it is unset, so seed it ONCE from the
        # presenting ST-elevation and troponin; thereafter evolve it by its own
        # dynamics below. This breaks the original self-amplifying loop in which
        # ischemia was re-derived from troponin/ST every step (whose own targets
        # were ~2.5x ischemia), driving every twin to full occlusion.
        ischemia = current_state.get('_ischemia_severity', None)
        if ischemia is None:
            # Seed latent ischemia from presentation. ST-elevation is the
            # immediate marker of the acute ischemic insult; troponin is a slower-
            # rising downstream marker, so it is down-weighted at presentation.
            # This also avoids slamming every higher-acuity presentation to a
            # clipped 1.0 (which removed all intra-group infarct-size variation).
            ischemia = np.clip(st_elevation / 3.6 + troponin / 40.0, 0, 1)
        ischemia = float(ischemia)

        # Intervention effects
        aspirin_effect = 0.0
        nitrate_effect = 0.0
        beta_blocker_effect = 0.0
        pci_effect = 0.0

        if interventions:
            if interventions.get('aspirin', False):
                aspirin_effect = 0.2
            if interventions.get('nitrate', False):
                nitrate_effect = 0.3
            if interventions.get('beta_blocker', False):
                beta_blocker_effect = 0.4
            if interventions.get('pci', False):
                pci_effect = 0.8  # Major improvement

        # Latent-ischemia dynamics (decoupled from the troponin/ST readouts).
        # Unreperfused vessels worsen slowly toward full occlusion; reperfusion
        # (PCI, and to a lesser degree nitrate) actively lowers ischemia. Only the
        # presence of a non-trivial ischemic insult drives progression, so a near-
        # normal presentation does NOT spuriously climb to occlusion.
        reperfusion = pci_effect + 0.3 * nitrate_effect
        d_ischemia = (
            self.ischemia_progression * ischemia * (1.0 - ischemia)  # natural hist.
            - reperfusion * ischemia                                 # treatment
        ) * dt
        ischemia = float(np.clip(ischemia + d_ischemia, 0.0, 1.0))

        # Accrue irreversible infarct, weighted by the decaying salvage window.
        # dD_inf = k_salvage * ischemia * exp(-lambda * t) * dt, clamped >= 0 so
        # the infarct is monotone non-decreasing. Earlier reperfusion lowers
        # ischemia while the salvage weight is still large, sparing more tissue.
        salvage_weight = float(np.exp(-self.salvage_lambda * elapsed))
        d_infarct = self.k_salvage * ischemia * salvage_weight * dt
        infarct = infarct + max(d_infarct, 0.0)
        elapsed_new = elapsed + dt

        # Troponin (rises with ongoing ischemia, falls with reperfusion)
        trop_target = 0.01 + 15.0 * ischemia - 10.0 * pci_effect
        d_trop = (trop_target - troponin) * 0.15 * dt
        trop_new = troponin + d_trop + rng.normal(0, 0.5 * dt)
        trop_new = np.clip(trop_new, 0.01, 50.0)

        # ST elevation
        st_target = 0.0 + 3.0 * ischemia - 2.5 * pci_effect
        d_st = (st_target - st_elevation) * 0.3 * dt
        st_new = st_elevation + d_st + rng.normal(0, 0.2 * dt)
        st_new = np.clip(st_new, 0.0, 5.0)

        # Chest pain
        pain_target = 0 + 8 * ischemia - 5 * nitrate_effect - 7 * pci_effect
        d_pain = (pain_target - chest_pain) * 0.5 * dt
        pain_new = chest_pain + d_pain + rng.normal(0, 0.5 * dt)
        pain_new = np.clip(pain_new, 0, 10)

        # Heart rate (reduced by beta blocker)
        hr_target = 75 + 25 * ischemia - 20 * beta_blocker_effect
        d_hr = (hr_target - hr) * 0.4 * dt
        hr_new = hr + d_hr + rng.normal(0, 3 * dt)
        hr_new = np.clip(hr_new, 45, 140)

        # Blood pressure (reduced by nitrate, beta blocker)
        bp_sys_target = (
            130 + 20 * ischemia - 25 * nitrate_effect - 20 * beta_blocker_effect
        )
        d_bp_sys = (bp_sys_target - bp_sys) * 0.3 * dt
        bp_sys_new = bp_sys + d_bp_sys + rng.normal(0, 5 * dt)
        bp_sys_new = np.clip(bp_sys_new, 80, 200)

        bp_dia_target = (
            80 + 10 * ischemia - 15 * nitrate_effect - 10 * beta_blocker_effect
        )
        d_bp_dia = (bp_dia_target - bp_dia) * 0.3 * dt
        bp_dia_new = bp_dia + d_bp_dia + rng.normal(0, 3 * dt)
        bp_dia_new = np.clip(bp_dia_new, 50, 120)

        # Update state
        next_state = current_state.copy()
        next_state.update(
            {
                'heart_rate': float(hr_new),
                'blood_pressure_sys': float(bp_sys_new),
                'blood_pressure_dia': float(bp_dia_new),
                'troponin': float(trop_new),
                'st_elevation': float(st_new),
                'chest_pain_score': float(pain_new),
                '_ischemia_severity': float(ischemia),
                # Cumulative irreversible infarct + elapsed clock (latent).
                '_infarct': float(infarct),
                'infarct': float(infarct),
                '_elapsed_hours': float(elapsed_new),
            }
        )

        return next_state
