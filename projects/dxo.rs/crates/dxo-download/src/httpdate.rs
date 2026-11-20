use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// 解析 `Retry-After`：delta-seconds 或 IMF-fixdate（合同 43 §4 要求遵守）。
pub(crate) fn parse_retry_after(value: &str, now: SystemTime) -> Option<Duration> {
    let value = value.trim();
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }
    let unix = parse_http_date(value)?;
    let now_unix = now.duration_since(UNIX_EPOCH).ok()?.as_secs() as i64;
    Some(Duration::from_secs(unix.saturating_sub(now_unix).max(0) as u64))
}

/// 解析 `Sun, 06 Nov 1994 08:49:37 GMT` 形式的 HTTP 日期为 Unix 秒。
fn parse_http_date(value: &str) -> Option<i64> {
    let (_weekday, rest) = value.split_once(", ")?;
    let mut parts = rest.split_whitespace();
    let day: u64 = parts.next()?.parse().ok()?;
    let month = parts.next()?;
    let year: i64 = parts.next()?.parse().ok()?;
    let time = parts.next()?;
    if parts.next().is_some_and(|zone| !zone.eq_ignore_ascii_case("GMT")) {
        return None;
    }
    let month = match month.to_ascii_lowercase().as_str() {
        "jan" => 1,
        "feb" => 2,
        "mar" => 3,
        "apr" => 4,
        "may" => 5,
        "jun" => 6,
        "jul" => 7,
        "aug" => 8,
        "sep" => 9,
        "oct" => 10,
        "nov" => 11,
        "dec" => 12,
        _ => return None,
    };
    let mut hms = time.split(':');
    let hour: u64 = hms.next()?.parse().ok()?;
    let minute: u64 = hms.next()?.parse().ok()?;
    let second: u64 = hms.next()?.parse().ok()?;
    if hms.next().is_some() || day == 0 || day > 31 || hour > 23 || minute > 59 || second > 60 {
        return None;
    }
    let days = days_from_civil(year, month as i64, day as i64);
    Some(days * 86_400 + hour as i64 * 3600 + minute as i64 * 60 + second as i64)
}

/// Howard Hinnant 的 civil-from-days 逆运算。
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let month_shifted = (month + 9) % 12;
    let day_of_year = (153 * month_shifted + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn parses_delta_seconds() {
        assert_eq!(parse_retry_after("7", SystemTime::now()), Some(Duration::from_secs(7)));
    }

    #[test]
    fn parses_imf_fixdate() {
        // Sun, 06 Nov 1994 08:49:37 GMT == 784111777
        let now = UNIX_EPOCH + Duration::from_secs(784111700);
        assert_eq!(parse_retry_after("Sun, 06 Nov 1994 08:49:37 GMT", now), Some(Duration::from_secs(77)));
    }

    #[test]
    fn rejects_garbage() {
        assert_eq!(parse_retry_after("soon", SystemTime::now()), None);
        assert_eq!(days_from_civil(1970, 1, 1), 0);
    }
}
