
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

try:
    import pymc as pm
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False

class HierarchicalRandomForest:
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs
        self.patient_rf = RandomForestRegressor(
            n_estimators=100, max_depth=15, min_samples_leaf=5,
            random_state=42, n_jobs=n_jobs
        )
        self.hospital_rf = RandomForestRegressor(
            n_estimators=75, max_depth=12, min_samples_leaf=8,
            random_state=42, n_jobs=n_jobs
        )
        self.region_rf = RandomForestRegressor(
            n_estimators=50, max_depth=10, min_samples_leaf=10,
            random_state=42, n_jobs=n_jobs
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X, y, groups):
        X_scaled = self.scaler.fit_transform(X)
        
        # Level 1: Patient features
        self.patient_rf.fit(X_scaled, y)
        patient_pred = self.patient_rf.predict(X_scaled)
        
        # Level 2: Hospital residuals
        hospital_residuals = y - patient_pred
        hospital_features = np.column_stack([X_scaled, groups['HOSP_NIS']])
        self.hospital_rf.fit(hospital_features, hospital_residuals)
        hospital_pred = self.hospital_rf.predict(hospital_features)
        
        # Level 3: Region residuals
        region_residuals = hospital_residuals - hospital_pred
        region_features = np.column_stack([X_scaled, groups['HOSP_NIS'], groups['HOSP_REGION']])
        self.region_rf.fit(region_features, region_residuals)
        
        self.is_fitted = True
        return self
    
    def predict(self, X, groups):
        if not self.is_fitted:
            raise NotFittedError("Call fit() first")
        
        X_scaled = self.scaler.transform(X)
        patient_pred = self.patient_rf.predict(X_scaled)
        hospital_pred = self.hospital_rf.predict(
            np.column_stack([X_scaled, groups['HOSP_NIS']])
        )
        region_pred = self.region_rf.predict(
            np.column_stack([X_scaled, groups['HOSP_NIS'], groups['HOSP_REGION']])
        )
        return patient_pred + hospital_pred + region_pred

class BayesianHRF:
    def __init__(self, n_trees=50, max_depth=10):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.scaler = StandardScaler()
        self.trace = None
        self.fallback_mode = not HAS_PYMC
        self.rf = None
        self.hospital_map = {}
        self.region_map = {}
    
    def fit(self, X, y, groups, n_samples=250, tune=500):
        X_scaled = self.scaler.fit_transform(X)
        
        self.rf = RandomForestRegressor(
            n_estimators=self.n_trees, max_depth=self.max_depth,
            n_jobs=-1, random_state=42
        )
        self.rf.fit(X_scaled, y)
        
        if self.fallback_mode:
            return self
        
        rf_preds = self.rf.predict(X_scaled)
        
        try:
            # Map groups to indices
            unique_hospitals = np.unique(groups['HOSP_NIS'])
            unique_regions = np.unique(groups['HOSP_REGION'])
            self.hospital_map = {h: i for i, h in enumerate(unique_hospitals)}
            self.region_map = {r: i for i, r in enumerate(unique_regions)}
            
            hosp_idx = np.array([self.hospital_map[h] for h in groups['HOSP_NIS']])
            region_idx = np.array([self.region_map[r] for r in groups['HOSP_REGION']])
            
            n_hospitals = len(unique_hospitals)
            n_regions = len(unique_regions)
            
            with pm.Model() as self.model:
                sigma = pm.HalfNormal('sigma', sigma=5)
                sigma_region = pm.HalfNormal('sigma_region', sigma=0.5)
                sigma_hospital = pm.HalfNormal('sigma_hospital', sigma=0.3)
                
                region_effects = pm.Normal('region_effects', mu=0, sigma=sigma_region, shape=n_regions)
                hospital_effects = pm.Normal('hospital_effects', mu=0, sigma=sigma_hospital, shape=n_hospitals)
                
                rf_bias = pm.Normal('rf_bias', mu=0, sigma=3)
                rf_scale = pm.HalfNormal('rf_scale', sigma=0.5)
                
                mu = (rf_bias + rf_scale * rf_preds + 
                      hospital_effects[hosp_idx] + region_effects[region_idx])
                y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
                
                self.trace = pm.sample(n_samples, tune=tune, cores=2, 
                                       target_accept=0.95, return_inferencedata=True)
        except Exception as e:
            print(f"PyMC failed: {e}. Using RF fallback.")
            self.fallback_mode = True
        
        return self
    
    def predict(self, X, groups, ci=0.95):
        X_scaled = self.scaler.transform(X)
        rf_preds = self.rf.predict(X_scaled)
        
        if self.fallback_mode or self.trace is None:
            se = np.std(rf_preds) * 0.5
            return rf_preds, rf_preds - 1.96*se, rf_preds + 1.96*se, None
        
        try:
            posterior = self.trace.posterior
            rf_bias = posterior["rf_bias"].values.flatten()
            rf_scale = posterior["rf_scale"].values.flatten()
            hospital_effects = posterior["hospital_effects"].values
            region_effects = posterior["region_effects"].values
            
            hosp_idx = np.array([self.hospital_map.get(h, 0) for h in groups['HOSP_NIS']])
            region_idx = np.array([self.region_map.get(r, 0) for r in groups['HOSP_REGION']])
            
            n_samples = len(rf_bias)
            n_test = len(X_scaled)
            all_preds = np.zeros((n_samples, n_test))
            
            for i in range(n_samples):
                h_eff = hospital_effects.reshape(-1, hospital_effects.shape[-1])[i, hosp_idx]
                r_eff = region_effects.reshape(-1, region_effects.shape[-1])[i, region_idx]
                all_preds[i] = rf_bias[i] + rf_scale[i] * rf_preds + h_eff + r_eff
            
            point = np.mean(all_preds, axis=0)
            lower = np.percentile(all_preds, (1-ci)/2 * 100, axis=0)
            upper = np.percentile(all_preds, (1+ci)/2 * 100, axis=0)
            
            return point, lower, upper, all_preds
        except Exception as e:
            print(f"Prediction error: {e}. Using fallback.")
            se = np.std(rf_preds) * 0.5
            return rf_preds, rf_preds - 1.96*se, rf_preds + 1.96*se, None

class ConformalHRF:
    
    def __init__(self, base_model=None, method='cdf_pooling'):
        self.base_model = base_model or HierarchicalRandomForest()
        self.method = method
        self.n_bootstrap = 50  # for repeated subsampling
        self.threshold = None
        self.is_calibrated = False
    
    def fit(self, X, y, groups):
        self.base_model.fit(X, y, groups)
        return self
    
    def _subsample_one_per_hospital(self, X, y, groups, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        hospitals = groups['HOSP_NIS']
        unique_hospitals = np.unique(hospitals)
        
        selected = []
        for hosp in unique_hospitals:
            hosp_indices = np.where(hospitals == hosp)[0]
            selected.append(np.random.choice(hosp_indices))
        
        idx = np.array(selected)
        groups_sub = {k: v[idx] for k, v in groups.items()}
        return X[idx], y[idx], groups_sub
    
    def _compute_threshold(self, scores, alpha):
        n = len(scores)
        q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        return np.quantile(scores, q_level)
    
    def calibrate(self, X_calib, y_calib, groups_calib, alpha=0.05):
        self.alpha = alpha
        
        if self.method == 'cdf_pooling':
            # Use all data — assumes exchangeability
            preds = self.base_model.predict(X_calib, groups_calib)
            scores = np.abs(y_calib - preds)
            self.threshold = self._compute_threshold(scores, alpha)
            
        elif self.method == 'subsampling_once':
            # One patient per hospital — breaks cluster dependence
            X_sub, y_sub, g_sub = self._subsample_one_per_hospital(
                X_calib, y_calib, groups_calib, seed=42
            )
            preds = self.base_model.predict(X_sub, g_sub)
            scores = np.abs(y_sub - preds)
            self.threshold = self._compute_threshold(scores, alpha)
            
        elif self.method == 'repeated_subsampling':
            # Multiple subsampling iterations — most robust
            thresholds = []
            for b in range(self.n_bootstrap):
                X_sub, y_sub, g_sub = self._subsample_one_per_hospital(
                    X_calib, y_calib, groups_calib, seed=42+b
                )
                preds = self.base_model.predict(X_sub, g_sub)
                scores = np.abs(y_sub - preds)
                thresholds.append(self._compute_threshold(scores, alpha))
            self.threshold = np.median(thresholds)
            self.all_thresholds = thresholds
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.is_calibrated = True
        return self
    
    def predict(self, X, groups):
        if not self.is_calibrated:
            raise RuntimeError("Call calibrate() first")
        
        preds = self.base_model.predict(X, groups)
        lower = preds - self.threshold
        upper = preds + self.threshold
        
        return preds, lower, upper

# Hybrid Bayesian-Conformal HRF (adaptive intervals + coverage guarantee)

class HybridConformalHRF:
    def __init__(self, method='cdf_pooling', gamma=1.0):
        self.hrf = HierarchicalRandomForest()
        self.bayesian = BayesianHRF()
        self.method = method
        self.n_bootstrap = 50
        self.gamma = gamma  # uncertainty scaling power
        self.threshold = None
        self.uncertainty_scale = None
        self.is_calibrated = False
    
    def fit(self, X, y, groups):
        self.hrf.fit(X, y, groups)
        self.bayesian.fit(X, y, groups)
        return self
    
    def _get_uncertainties(self, X, groups):
        _, _, _, all_preds = self.bayesian.predict(X, groups)
        if all_preds is not None:
            return np.std(all_preds, axis=0)
        else:
            # Fallback: constant uncertainty
            return np.ones(len(X))
    
    def _subsample_one_per_hospital(self, X, y, groups, uncertainties, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        hospitals = groups['HOSP_NIS']
        unique_hospitals = np.unique(hospitals)
        
        selected = []
        for hosp in unique_hospitals:
            hosp_indices = np.where(hospitals == hosp)[0]
            selected.append(np.random.choice(hosp_indices))
        
        idx = np.array(selected)
        groups_sub = {k: v[idx] for k, v in groups.items()}
        return X[idx], y[idx], groups_sub, uncertainties[idx]
    
    def _compute_weighted_threshold(self, residuals, uncertainties, alpha):
        # Scale uncertainties
        scaled_unc = np.power(uncertainties, self.gamma)
        scaled_unc = np.maximum(scaled_unc, 1e-6)  # avoid division by zero
        
        # Weight scores by inverse uncertainty
        weighted_scores = np.abs(residuals) / scaled_unc
        
        n = len(weighted_scores)
        q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        return np.quantile(weighted_scores, q_level)
    
    def calibrate(self, X_calib, y_calib, groups_calib, alpha=0.05):
        self.alpha = alpha
        
        # Get predictions and uncertainties
        preds = self.hrf.predict(X_calib, groups_calib)
        uncertainties = self._get_uncertainties(X_calib, groups_calib)
        residuals = y_calib - preds
        
        # Store uncertainty scale for prediction time
        self.uncertainty_scale = np.mean(uncertainties)
        
        if self.method == 'cdf_pooling':
            self.threshold = self._compute_weighted_threshold(residuals, uncertainties, alpha)
            
        elif self.method == 'subsampling_once':
            X_sub, y_sub, g_sub, unc_sub = self._subsample_one_per_hospital(
                X_calib, y_calib, groups_calib, uncertainties, seed=42
            )
            preds_sub = self.hrf.predict(X_sub, g_sub)
            residuals_sub = y_sub - preds_sub
            self.threshold = self._compute_weighted_threshold(residuals_sub, unc_sub, alpha)
            
        elif self.method == 'repeated_subsampling':
            thresholds = []
            for b in range(self.n_bootstrap):
                X_sub, y_sub, g_sub, unc_sub = self._subsample_one_per_hospital(
                    X_calib, y_calib, groups_calib, uncertainties, seed=42+b
                )
                preds_sub = self.hrf.predict(X_sub, g_sub)
                residuals_sub = y_sub - preds_sub
                thresholds.append(self._compute_weighted_threshold(residuals_sub, unc_sub, alpha))
            self.threshold = np.median(thresholds)
            self.all_thresholds = thresholds
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.is_calibrated = True
        return self
    
    def predict(self, X, groups):
        if not self.is_calibrated:
            raise RuntimeError("Call calibrate() first")
        
        preds = self.hrf.predict(X, groups)
        uncertainties = self._get_uncertainties(X, groups)
        
        # Scale uncertainties same way as calibration
        scaled_unc = np.power(uncertainties, self.gamma)
        scaled_unc = np.maximum(scaled_unc, 1e-6)
        
        # Adaptive interval width
        interval_width = self.threshold * scaled_unc
        
        lower = preds - interval_width
        upper = preds + interval_width
        
        return preds, lower, upper, uncertainties



def create_all_models(method='cdf_pooling'):
    
    return {
        'hrf': HierarchicalRandomForest(),
        'bayesian_hrf': BayesianHRF(),
        'conformal_hrf': ConformalHRF(method=method),
        'hybrid_conformal_hrf': HybridConformalHRF(method=method)
    }

