##fateserver project facts:

This repository is the source of https://fate.ffmpeg.org, the web interface showing FATE (FFmpeg's test suite) results submitted by client machines.
It is a set of Perl CGI scripts (index.cgi, history.cgi, log.cgi, report.cgi) sharing FATE.pm, plus fate-recv.sh which receives report uploads (a tar of report + logs, one directory per slot under $FATEDIR).
The CGI scripts run under Perl taint mode (-T) and parse attacker-suppliable report files and query parameters; input handling and escaping (HTML::Entities, URI::Escape) are security-relevant.
Report format compatibility matters: existing stored reports and deployed FATE clients keep submitting in the old formats, so parsers must stay backward compatible (see the version field in the report header).
There is no test suite in this repository; changes are verified by running the CGI scripts against real stored reports.
