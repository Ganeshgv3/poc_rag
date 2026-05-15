-- phpMyAdmin SQL Dump
-- version 5.2.1deb3
-- https://www.phpmyadmin.net/
--
-- Host: localhost:3306
-- Generation Time: May 15, 2026 at 06:59 AM
-- Server version: 8.0.45-0ubuntu0.24.04.1
-- PHP Version: 8.4.21

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `rag_app`
--

-- --------------------------------------------------------

--
-- Table structure for table `chats`
--

CREATE TABLE `chats` (
  `id` bigint NOT NULL,
  `user_id` bigint NOT NULL,
  `document_id` bigint NOT NULL,
  `title` varchar(255) NOT NULL DEFAULT 'New Chat',
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `deleted_at` timestamp NULL DEFAULT NULL,
  `pinned_at` timestamp NULL DEFAULT NULL,
  `archived_at` timestamp NULL DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `chats`
--

INSERT INTO `chats` (`id`, `user_id`, `document_id`, `title`, `created_at`, `deleted_at`, `pinned_at`, `archived_at`) VALUES
(40, 1, 16, 'Personal Profile & Career Experience', '2026-05-14 08:21:41', NULL, NULL, NULL),
(41, 1, 17, '2009 Bob Completed', '2026-05-14 08:37:10', NULL, NULL, NULL),
(42, 1, 17, '2009 Africa Average', '2026-05-14 08:42:21', NULL, NULL, NULL),
(43, 1, 16, 'Career Experience', '2026-05-14 14:11:03', NULL, NULL, NULL),
(49, 1, 19, 'Any How Load', '2026-05-15 05:46:08', NULL, NULL, NULL);

-- --------------------------------------------------------

--
-- Table structure for table `documents`
--

CREATE TABLE `documents` (
  `id` bigint NOT NULL,
  `user_id` bigint NOT NULL,
  `filename` varchar(255) NOT NULL,
  `stored_name` varchar(255) NOT NULL,
  `path` text NOT NULL,
  `sha256` varchar(64) NOT NULL,
  `collection_name` varchar(120) NOT NULL,
  `chunks` int NOT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `documents`
--

INSERT INTO `documents` (`id`, `user_id`, `filename`, `stored_name`, `path`, `sha256`, `collection_name`, `chunks`, `created_at`) VALUES
(16, 1, 'Jan 2026 Pay Slip.pdf', '3710718f8174_Jan 2026 Pay Slip.pdf', '/var/www/php83-app/poc_rag/data/uploads/3710718f8174_Jan 2026 Pay Slip.pdf', '3710718f8174c22ea60c46ff5730716585ed46f83f0fab45d181589b870dbe5b', 'pdf_3710718f8174c22e', 1, '2026-05-14 08:21:31'),
(17, 1, 'sample-tables.pdf', '5746428c7c3d_sample-tables.pdf', '/var/www/php83-app/poc_rag/data/uploads/5746428c7c3d_sample-tables.pdf', '5746428c7c3d6225ae6a043ed6f6566ccdd42d59471830ca9428faf2254a4220', 'pdf_5746428c7c3d6225', 11, '2026-05-14 08:34:52'),
(19, 1, 'MSRlive! 2026 User Guide.pdf', '675021d6e1a6_MSRlive! 2026 User Guide.pdf', '/var/www/php83-app/poc_rag/data/uploads/675021d6e1a6_MSRlive! 2026 User Guide.pdf', '675021d6e1a65e092e94787474daa0bcb43c9dbf68d96b68bf773e46aff75f44', 'pdf_675021d6e1a65e09', 18, '2026-05-15 05:45:56');

-- --------------------------------------------------------

--
-- Table structure for table `messages`
--

CREATE TABLE `messages` (
  `id` bigint NOT NULL,
  `chat_id` bigint NOT NULL,
  `role` enum('user','assistant') NOT NULL,
  `content` mediumtext NOT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `retrieval_seconds` double DEFAULT NULL,
  `latency_seconds` double DEFAULT NULL,
  `accuracy_label` varchar(24) DEFAULT NULL,
  `accuracy_score` int DEFAULT NULL,
  `retrieval_context_json` json DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `messages`
--

INSERT INTO `messages` (`id`, `chat_id`, `role`, `content`, `created_at`, `retrieval_seconds`, `latency_seconds`, `accuracy_label`, `accuracy_score`, `retrieval_context_json`) VALUES
(403, 40, 'user', 'hi', '2026-05-14 08:21:41', NULL, NULL, NULL, NULL, NULL),
(404, 40, 'assistant', 'Hi there. I am ready to help. Ask me anything from your selected PDF.', '2026-05-14 08:21:41', NULL, NULL, NULL, NULL, NULL),
(405, 40, 'user', 'Name?', '2026-05-14 08:22:00', NULL, NULL, NULL, NULL, NULL),
(406, 40, 'assistant', 'Ganesh Moorthy R', '2026-05-14 08:22:02', NULL, NULL, NULL, NULL, NULL),
(407, 40, 'user', 'company details?', '2026-05-14 08:22:14', NULL, NULL, NULL, NULL, NULL),
(408, 40, 'assistant', 'Galaxy Weblinks Ltd (Sivakasi) \n1496 - P.K.N Road, 1st floor, Sivakasi', '2026-05-14 08:22:15', NULL, NULL, NULL, NULL, NULL),
(409, 40, 'user', 'bank statement?', '2026-05-14 08:22:37', NULL, NULL, NULL, NULL, NULL),
(410, 40, 'assistant', 'The PDF does not contain a bank statement.', '2026-05-14 08:22:38', NULL, NULL, NULL, NULL, NULL),
(411, 40, 'user', 'bank name?', '2026-05-14 08:22:55', NULL, NULL, NULL, NULL, NULL),
(412, 40, 'assistant', 'Bank of India', '2026-05-14 08:22:56', NULL, NULL, NULL, NULL, NULL),
(413, 40, 'user', 'woking days', '2026-05-14 08:23:20', NULL, NULL, NULL, NULL, NULL),
(414, 40, 'assistant', '31 days', '2026-05-14 08:23:21', NULL, NULL, NULL, NULL, NULL),
(415, 40, 'user', 'does Basic Pay credited?', '2026-05-14 08:23:42', NULL, NULL, NULL, NULL, NULL),
(416, 40, 'assistant', 'Yes.', '2026-05-14 08:23:42', NULL, NULL, NULL, NULL, NULL),
(417, 41, 'user', 'how much Bob completed in 2009', '2026-05-14 08:37:10', NULL, NULL, NULL, NULL, NULL),
(418, 41, 'assistant', 'I am here to help. Could you please rephrase your question?', '2026-05-14 08:37:12', NULL, NULL, NULL, NULL, NULL),
(419, 41, 'user', 'Bob completed in 2009', '2026-05-14 08:37:26', NULL, NULL, NULL, NULL, NULL),
(420, 41, 'assistant', '19', '2026-05-14 08:37:27', NULL, NULL, NULL, NULL, NULL),
(421, 42, 'user', 'how much Sue completed in 2009', '2026-05-14 08:42:21', NULL, NULL, NULL, NULL, NULL),
(422, 42, 'assistant', '10', '2026-05-14 08:42:31', NULL, NULL, NULL, NULL, NULL),
(423, 42, 'user', 'what is the average rainfall in Africa', '2026-05-14 08:43:46', NULL, NULL, NULL, NULL, NULL),
(424, 42, 'assistant', 'I am here to help. Could you please rephrase your question?', '2026-05-14 08:43:47', NULL, NULL, NULL, NULL, NULL),
(425, 43, 'user', 'work mode?', '2026-05-14 14:11:03', NULL, NULL, NULL, NULL, NULL),
(426, 43, 'assistant', 'WFH', '2026-05-14 14:11:12', NULL, NULL, NULL, NULL, NULL),
(427, 43, 'user', 'rtsd', '2026-05-14 14:11:34', NULL, NULL, NULL, NULL, NULL),
(428, 43, 'assistant', 'I am here to help. Could you please rephrase your question?', '2026-05-14 14:11:35', NULL, NULL, NULL, NULL, NULL),
(429, 43, 'user', 'dsfs', '2026-05-14 14:13:46', NULL, NULL, NULL, NULL, NULL),
(430, 43, 'assistant', 'I’m sorry, but I can’t find any information related to that in the document.', '2026-05-14 14:13:55', NULL, NULL, NULL, NULL, NULL),
(431, 43, 'user', 'dfgdfvdfbvdf dfb', '2026-05-14 14:14:42', NULL, NULL, NULL, NULL, NULL),
(432, 43, 'assistant', 'I’m sorry, I couldn’t find an answer to that in the document.', '2026-05-14 14:14:51', NULL, NULL, NULL, NULL, NULL),
(433, 43, 'user', 'when did the salary credited?', '2026-05-14 14:15:20', NULL, NULL, NULL, NULL, NULL),
(434, 43, 'assistant', 'The pay slip does not specify the exact date the salary was credited; it is for January 2026.', '2026-05-14 14:15:22', NULL, NULL, NULL, NULL, NULL),
(435, 42, 'user', 'I am here to help. Could you please rephrase your question?', '2026-05-14 14:19:39', NULL, NULL, NULL, NULL, NULL),
(436, 42, 'assistant', 'The PDF did not surface a clear match for Am Help. Rephrase Question this time. Ask with a short phrase you can see on the page—an amount, heading, or date works well.', '2026-05-14 14:19:40', NULL, NULL, NULL, NULL, NULL),
(513, 49, 'user', 'hi', '2026-05-15 05:46:08', NULL, NULL, NULL, NULL, NULL),
(514, 49, 'assistant', 'Hi there. I am ready to help. Ask me anything from your selected PDF.', '2026-05-15 05:46:08', 0, 0.01, 'N/A', 0, NULL),
(515, 49, 'user', 'is there any option to load portfolio', '2026-05-15 05:46:16', NULL, NULL, NULL, NULL, NULL),
(516, 49, 'assistant', 'Yes.', '2026-05-15 05:46:17', 1.7565, 0.01, 'Medium', 78, '[\"Portfolio Name, then click Run.\\nYou will ultimately arrive at the Value a Portfolio screen with both approaches.\\n\\nMSRLIVE! USER GUIDE\\n\\n14\\nYour Trusted Capital Markets Partner\\n\\nUnder Choose Portfolio, select the specific portfolio you wish to value. Under Input Templates, the Market input template is the standard MCT template, but you can always select a customized template if preferred. The Reference Table\\nTemplate is set to a Default template. The Reference Tables consist of the prepayment rates and other key internal variables. They are visible, but not modifiable. If you would like to develop a custom template for reference tables, please contact the MCT MSR Team.\\nNote: You can view any of the selected portfolio options by clicking on the View button next to the drop-down. It will show you the portfolio history, or the selected templates.\\nThe fields shown under Key Economic Sensitivities are the most commonly changed Primary Input fields.\", \"go to the Settings tab and select File Formats.\\n\\nMSRLIVE! USER GUIDE\\n\\n4\\nYour Trusted Capital Markets Partner\\nThen click +Create in the top right-hand corner to get started.\\n\\nUPLOAD FILE\\nUpload a csv file of the portfolio format by clicking the blue upload button. It is recommended to upload a file that only contains the headers (no data) and one loan with actual data as a sample for the file format set up. Enter a name for the File Format and then select Next.\\n\\nMSRLIVE! USER GUIDE\\n\\n5\\nYour Trusted Capital Markets Partner\\nDATA MAPPING\\nThe model will then automatically map the fields in the portfolio and will choose the best match for each category in the portfolio. The populated value comes from the uploaded portfolio and shows in parentheses the first value in the data. There are three pages of fields that need to be verified before proceeding. If a populated value is incorrect, the drop down can be used to modify the value to the correct field.\", \"be revealed.\\n\\nMSRLIVE! USER GUIDE\\n\\n17\\nYour Trusted Capital Markets Partner\\n3. RUNNNING RATE SHOCK ANALYSIS\\nOn the Dashboard, select Run Rate Shock Analysis then click Start.\\n\\nAs previously outlined, just like when completing a portfolio valuation, you need to select the portfolio name, the input template, and reference table that you want to use. The Create a Repline section will automatically be set to yes, which will run the Rate Shocks by grouping the portfolio into similar tranches. Please note, you can run the Hedge valuation without replines on a loan level basis, but it will take significantly longer. Once the templates have been selected click\\nRun.\\n\\nMSRLIVE! USER GUIDE\\n\\n18\\nYour Trusted Capital Markets Partner\\nThe Recent Activities section will update, and you can click on the red progress circle to watch the progress.\\n\\nOnce complete, you can either click the Hedge Valuation complete pop-up, or, go to Portfolios, select the portfolio name, and it will be in added to the list where Hedge is listed as Yes.\\n\\nClick on View Details.\\n\\nMSRLIVE! USER GUIDE\"]'),
(517, 49, 'user', 'how?', '2026-05-15 05:46:24', NULL, NULL, NULL, NULL, NULL),
(518, 49, 'assistant', 'To load a portfolio, go to the Settings tab, choose File Formats, click +Create, then upload a CSV file of the portfolio format by clicking the blue upload button. After uploading, give the file format a name and click Next. The model will map the fields, allowing you to verify and adjust them before proceeding.', '2026-05-15 05:46:29', 1.5623, 3.89, 'Medium', 68, '[\"be revealed.\\n\\nMSRLIVE! USER GUIDE\\n\\n17\\nYour Trusted Capital Markets Partner\\n3. RUNNNING RATE SHOCK ANALYSIS\\nOn the Dashboard, select Run Rate Shock Analysis then click Start.\\n\\nAs previously outlined, just like when completing a portfolio valuation, you need to select the portfolio name, the input template, and reference table that you want to use. The Create a Repline section will automatically be set to yes, which will run the Rate Shocks by grouping the portfolio into similar tranches. Please note, you can run the Hedge valuation without replines on a loan level basis, but it will take significantly longer. Once the templates have been selected click\\nRun.\\n\\nMSRLIVE! USER GUIDE\\n\\n18\\nYour Trusted Capital Markets Partner\\nThe Recent Activities section will update, and you can click on the red progress circle to watch the progress.\\n\\nOnce complete, you can either click the Hedge Valuation complete pop-up, or, go to Portfolios, select the portfolio name, and it will be in added to the list where Hedge is listed as Yes.\\n\\nClick on View Details.\\n\\nMSRLIVE! USER GUIDE\", \"go to the Settings tab and select File Formats.\\n\\nMSRLIVE! USER GUIDE\\n\\n4\\nYour Trusted Capital Markets Partner\\nThen click +Create in the top right-hand corner to get started.\\n\\nUPLOAD FILE\\nUpload a csv file of the portfolio format by clicking the blue upload button. It is recommended to upload a file that only contains the headers (no data) and one loan with actual data as a sample for the file format set up. Enter a name for the File Format and then select Next.\\n\\nMSRLIVE! USER GUIDE\\n\\n5\\nYour Trusted Capital Markets Partner\\nDATA MAPPING\\nThe model will then automatically map the fields in the portfolio and will choose the best match for each category in the portfolio. The populated value comes from the uploaded portfolio and shows in parentheses the first value in the data. There are three pages of fields that need to be verified before proceeding. If a populated value is incorrect, the drop down can be used to modify the value to the correct field.\", \"Portfolio Name, then click Run.\\nYou will ultimately arrive at the Value a Portfolio screen with both approaches.\\n\\nMSRLIVE! USER GUIDE\\n\\n14\\nYour Trusted Capital Markets Partner\\n\\nUnder Choose Portfolio, select the specific portfolio you wish to value. Under Input Templates, the Market input template is the standard MCT template, but you can always select a customized template if preferred. The Reference Table\\nTemplate is set to a Default template. The Reference Tables consist of the prepayment rates and other key internal variables. They are visible, but not modifiable. If you would like to develop a custom template for reference tables, please contact the MCT MSR Team.\\nNote: You can view any of the selected portfolio options by clicking on the View button next to the drop-down. It will show you the portfolio history, or the selected templates.\\nThe fields shown under Key Economic Sensitivities are the most commonly changed Primary Input fields.\"]'),
(519, 49, 'user', 'how to Execute portfolio', '2026-05-15 05:46:45', NULL, NULL, NULL, NULL, NULL),
(520, 49, 'assistant', 'To run a portfolio in MSRlive!: \n1. Open the Dashboard. \n2. In the Value a Portfolio screen (or Run Rate Shock Analysis screen), select the portfolio you want', '2026-05-15 05:46:48', 1.6397, 1.89, 'Medium', 72, '[\"be revealed.\\n\\nMSRLIVE! USER GUIDE\\n\\n17\\nYour Trusted Capital Markets Partner\\n3. RUNNNING RATE SHOCK ANALYSIS\\nOn the Dashboard, select Run Rate Shock Analysis then click Start.\\n\\nAs previously outlined, just like when completing a portfolio valuation, you need to select the portfolio name, the input template, and reference table that you want to use. The Create a Repline section will automatically be set to yes, which will run the Rate Shocks by grouping the portfolio into similar tranches. Please note, you can run the Hedge valuation without replines on a loan level basis, but it will take significantly longer. Once the templates have been selected click\\nRun.\\n\\nMSRLIVE! USER GUIDE\\n\\n18\\nYour Trusted Capital Markets Partner\\nThe Recent Activities section will update, and you can click on the red progress circle to watch the progress.\\n\\nOnce complete, you can either click the Hedge Valuation complete pop-up, or, go to Portfolios, select the portfolio name, and it will be in added to the list where Hedge is listed as Yes.\\n\\nClick on View Details.\\n\\nMSRLIVE! USER GUIDE\", \"Portfolio Name, then click Run.\\nYou will ultimately arrive at the Value a Portfolio screen with both approaches.\\n\\nMSRLIVE! USER GUIDE\\n\\n14\\nYour Trusted Capital Markets Partner\\n\\nUnder Choose Portfolio, select the specific portfolio you wish to value. Under Input Templates, the Market input template is the standard MCT template, but you can always select a customized template if preferred. The Reference Table\\nTemplate is set to a Default template. The Reference Tables consist of the prepayment rates and other key internal variables. They are visible, but not modifiable. If you would like to develop a custom template for reference tables, please contact the MCT MSR Team.\\nNote: You can view any of the selected portfolio options by clicking on the View button next to the drop-down. It will show you the portfolio history, or the selected templates.\\nThe fields shown under Key Economic Sensitivities are the most commonly changed Primary Input fields.\", \"the template. These tables can be discussed in depth with the MCT MSR team and how they impact the valuation.\\n\\nMSRLIVE! USER GUIDE\\n\\n8\\nYour Trusted Capital Markets Partner\\n\\nMSRLIVE! USER GUIDE\\n\\n9\\nYour Trusted Capital Markets Partner\\n2. VALUING A PORTFOLIO\\nBefore we can begin valuing a portfolio, you must select your desired File Format and create your Input Templates.\\nOnce this has been completed, proceed to the Dashboard page of MSRlive!, select Load a New Portfolio and click\\nStart.\\n\\nUPLOAD A NEW PORTFOLIO\\nUpload a new portfolio to MSRlive! by clicking the blue upload button (select file), then enter a name for the portfolio and select the File Format created for this portfolio using the drop-down menu. Once all selections have been made, click LOAD.\\n\\nMSRLIVE! USER GUIDE\\n\\n10\\nYour Trusted Capital Markets Partner\\nTROUBLESHOOTING DATA ERRORS\\nThe model will then go through all the loans in the portfolio and check for any errors. If there are errors, they will be listed as shown below, with the error description and value.\"]');

-- --------------------------------------------------------

--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `id` bigint NOT NULL,
  `name` varchar(120) NOT NULL,
  `email` varchar(255) NOT NULL,
  `password_hash` varchar(255) NOT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `users`
--

INSERT INTO `users` (`id`, `name`, `email`, `password_hash`, `created_at`) VALUES
(1, 'Ganesh', 'gm@gmail.com', 'scrypt:32768:8:1$KiiO4B0SHeDoeb73$cda60cc8ad59ced4135c813160fd88156c312a612bc4f3f53bec6e493495f56f558dd32232f04f160dca68d802a9213ddfc3fbbda078e210c561e823c8eb0b95', '2026-05-06 11:50:31');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `chats`
--
ALTER TABLE `chats`
  ADD PRIMARY KEY (`id`),
  ADD KEY `user_id` (`user_id`),
  ADD KEY `document_id` (`document_id`);

--
-- Indexes for table `documents`
--
ALTER TABLE `documents`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `uniq_user_sha` (`user_id`,`sha256`);

--
-- Indexes for table `messages`
--
ALTER TABLE `messages`
  ADD PRIMARY KEY (`id`),
  ADD KEY `chat_id` (`chat_id`);

--
-- Indexes for table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `email` (`email`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `chats`
--
ALTER TABLE `chats`
  MODIFY `id` bigint NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=50;

--
-- AUTO_INCREMENT for table `documents`
--
ALTER TABLE `documents`
  MODIFY `id` bigint NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=20;

--
-- AUTO_INCREMENT for table `messages`
--
ALTER TABLE `messages`
  MODIFY `id` bigint NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=521;

--
-- AUTO_INCREMENT for table `users`
--
ALTER TABLE `users`
  MODIFY `id` bigint NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `chats`
--
ALTER TABLE `chats`
  ADD CONSTRAINT `chats_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE,
  ADD CONSTRAINT `chats_ibfk_2` FOREIGN KEY (`document_id`) REFERENCES `documents` (`id`) ON DELETE CASCADE;

--
-- Constraints for table `documents`
--
ALTER TABLE `documents`
  ADD CONSTRAINT `documents_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE;

--
-- Constraints for table `messages`
--
ALTER TABLE `messages`
  ADD CONSTRAINT `messages_ibfk_1` FOREIGN KEY (`chat_id`) REFERENCES `chats` (`id`) ON DELETE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
