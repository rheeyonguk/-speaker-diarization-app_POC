# 브랜드 자산 (로고 · 파비콘)

공식 CI 로고 파일을 이 폴더에 넣고 `config/default.yaml`(또는 `config/local.yaml`)에서 지정하세요.

```yaml
ui:
  brand:
    logo_path: assets/brand/logo.png       # PNG/SVG, 흰 배경 위에 표시됨, 2 MB 이하
    favicon_path: assets/brand/favicon.png # 브라우저 탭 아이콘
    primary_color: "#RRGGBB"               # 브랜드 가이드의 공식 메인 컬러
    accent_color: "#RRGGBB"                # 보조 컬러 (헤더 그라데이션 끝색 · 강조선)
```

- 로고를 지정하면 헤더의 회사명 배지 대신 로고 이미지가 표시됩니다.
- 기본 색상(`#0b3d91`, `#0091d5`)은 공식 CI 값이 아닌 임시값입니다.
- 공식 CI 파일 사용 범위(사내 시스템 사용 가능 여부)는 브랜드 담당 부서 기준을 따르세요.
