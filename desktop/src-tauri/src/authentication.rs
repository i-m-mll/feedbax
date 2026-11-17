use block2::RcBlock;
use objc2::runtime::Bool;
use objc2_foundation::{NSError, NSString};
use objc2_local_authentication::{LAContext, LAPolicy};
use serde::{Deserialize, Serialize};
use std::io::{self, Read};
use std::sync::mpsc;
use std::time::Duration;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Challenge {
    challenge: String,
    reason: String,
}

#[derive(Serialize)]
struct Attested {
    challenge: String,
}

fn respond(
    input: impl Read,
    authenticate: impl FnOnce(&str) -> crate::service::Result<()>,
) -> crate::service::Result<Attested> {
    let challenge: Challenge = serde_json::from_reader(input.take(4096))?;
    if challenge.challenge.len() != 64
        || !challenge
            .challenge
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        || challenge.reason.is_empty()
        || challenge.reason.chars().any(char::is_control)
        || challenge.reason.len() > 512
    {
        return Err("Invalid Studio machine approval challenge".into());
    }
    authenticate(&challenge.reason)?;
    Ok(Attested {
        challenge: challenge.challenge,
    })
}

fn touch_id(reason: &str) -> crate::service::Result<()> {
    let policy = LAPolicy::DeviceOwnerAuthenticationWithBiometrics;
    let (tx, rx) = mpsc::sync_channel(1);
    let reply = RcBlock::new(move |success: Bool, _: *mut NSError| {
        let _ = tx.send(success.as_bool());
    });
    let reason = NSString::from_str(reason);
    let context = unsafe { LAContext::new() };
    unsafe {
        context
            .canEvaluatePolicy_error(policy)
            .map_err(|_| "Touch ID is unavailable for machine approval")?;
        context.setLocalizedFallbackTitle(Some(&NSString::from_str("")));
        context.setTouchIDAuthenticationAllowableReuseDuration(0.0);
        context.evaluatePolicy_localizedReason_reply(policy, &reason, &reply);
    }
    let result = rx.recv_timeout(Duration::from_secs(45));
    unsafe { context.invalidate() };
    if result? {
        Ok(())
    } else {
        Err("Machine approval was declined".into())
    }
}

pub fn attest_machine() -> crate::service::Result<()> {
    let proof = respond(io::stdin().lock(), touch_id)?;
    serde_json::to_writer(io::stdout().lock(), &proof)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evidence_requires_one_valid_challenge_and_successful_authentication() {
        let challenge = "a1".repeat(32);
        let bytes = serde_json::to_vec(&serde_json::json!({
            "challenge": challenge,
            "reason": "Approve 2 runs",
        }))
        .unwrap();
        let proof = respond(bytes.as_slice(), |reason| {
            assert_eq!(reason, "Approve 2 runs");
            Ok(())
        })
        .unwrap();
        assert_eq!(proof.challenge, challenge);
        assert!(respond(bytes.as_slice(), |_| Err("User cancelled".into())).is_err());
        assert!(respond(
            br#"{"challenge":"bogus","reason":"Approve"}"#.as_slice(),
            |_| { panic!("Malformed challenge reached the authenticator") }
        )
        .is_err());
    }
}
