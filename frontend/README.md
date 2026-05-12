# Frontend

Bu klasor, veri madenciligi projesinin iki ekranli frontend demolarini barindirir.

## Amac

- Kullanici, dataset alanlarina gore kendi verisini kontrollu form alanlari ile girer.
- Sistem bu veriyi TypeScript icinde calisan teklif motoruyla yorumlar.
- Ikinci ekranda statik bir e-ticaret urun sayfasi ustunde indirim popup'i gosterilir.

## Komutlar

```bash
npm install
npm run dev
```

Production build ve test:

```bash
npm run test
npm run build
```

`npm run sync-data`, Python tarafinda uretildigi varsayilan Faz 8-11 artefaktlarini okuyup frontend icin gereken `project-context.json` dosyasini gunceller.
